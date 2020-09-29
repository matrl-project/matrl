import numpy as np
import tensorflow as tf
import utils.core as core

from spinup.utils.mpi_tools import (
    mpi_fork,
    mpi_avg,
    proc_id,
    mpi_statistics_scalar,
    num_procs,
)
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
import os


class PGAgent:
    # TODO: add real policy and temp policy
    def __init__(self, id, obs_space, action_space, config):
        self.config = config
        seed = config["seed"]
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.is_rnn = self.config["network"] == "rnn"
        self.is_cnn = self.config["network"] == "cnn"
        if self.is_rnn:
            self.network = core.rnn
        elif self.is_cnn:
            self.network = core.cnn
        else:
            self.network = core.mlp

        self.id = id

        # Share information about action space with policy architecture
        config["action_space"] = action_space

        with tf.name_scope("Agent_{}".format(self.id)):
            # Inputs to computation graph
            self.x_ph, self.a_ph = core.placeholders_from_spaces(obs_space, action_space)
            (
                adv_ph,
                ret_ph,
                self.logp_old_ph,
                self.importance_ratio_ph,
                self.val_ph,
            ) = core.placeholders(None, None, None, None, None)

            if self.is_rnn:
                self.x_ph = core.placeholder(
                    (
                        config["rnn_length"],
                        obs_space.shape[0] + config["num_agent"] * action_space.n,
                    )
                )
            # Main outputs from computation graph
            # main policy
            self.pi, logp, logp_pi, self.v = core.mlp_actor_critic(
                self.network, self.x_ph, self.a_ph, config, "origin_{}".format(self.id)
            )
            # tmp policy
            pi_tmp, logp_tmp, logp_pi_tmp, self.v_tmp = core.mlp_actor_critic(
                self.network, self.x_ph, self.a_ph, config, "tmp_{}".format(self.id)
            )

            # Need all placeholders in *this* order later (to zip with data from buffer)
            self.all_phs = [
                self.x_ph,
                self.a_ph,
                adv_ph,
                ret_ph,
                self.logp_old_ph,
                self.val_ph,
            ]
            # Every step, get: action, value, and logprob
            self.get_action_ops = [self.pi, self.v, logp_pi]
            self.get_tmp_action_ops = [pi_tmp, self.v_tmp, logp_pi_tmp]

            # Experience buffer
            self.config["local_steps_per_epoch"] = int(
                config["steps_per_epoch"] / num_procs()
            )

            # Count variables
            var_counts = tuple(core.count_vars(scope) for scope in ["pi", "v"])
            print("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

            # --------------------------PPO objectives---------------------------------------------
            self.ratio = tf.exp(logp - self.logp_old_ph)  # pi(a|s) / pi_old(a|s)

            self.ratio_clipped = tf.clip_by_value(
                self.ratio, 1 - config["clip_ratio"], 1 + config["clip_ratio"]
            )

            # Info (useful to watch during learning)
            self.approx_kl = tf.reduce_mean(
                self.logp_old_ph - logp
            )  # a sample estimate for KL-divergence, easy to compute
            self.old_logp = tf.reduce_mean(self.logp_old_ph)
            self.approx_ent = tf.reduce_mean(
                -logp
            )  # a sample estimate for entropy, also easy to compute
            self.logp = tf.reduce_mean(logp)
            self.adv_mean = tf.reduce_mean(adv_ph)

            # define loss for pure pg or ppo
            if self.config["algo"] == "ppo":
                self.pi_loss = -tf.reduce_mean(
                    tf.minimum(self.ratio_clipped * adv_ph, self.ratio * adv_ph)
                )
            else:
                self.pi_loss = -tf.reduce_mean(self.ratio * adv_ph)

            vf_loss1 = tf.square(self.v - ret_ph)
            vf_clipped = self.val_ph + tf.clip_by_value(
                self.v - self.val_ph, -config["vf_clip_param"], config["vf_clip_param"]
            )
            vf_loss2 = tf.square(vf_clipped - ret_ph)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = tf.reduce_mean(vf_loss)

            # self.v_loss = tf.reduce_mean((ret_ph - self.v) ** 2) * self.config["v_loss_coeff"]
            self.loss = (
                self.pi_loss * self.config["pi_loss_coeff"]
                + self.mean_vf_loss * self.config["v_loss_coeff"]
                - self.config["entropy_coeff"] * self.approx_ent
                + self.config["kl_coeff"] * self.approx_kl
            )

            # best response loss
            importance_ratio = tf.reduce_mean(
                tf.clip_by_value(
                    self.importance_ratio_ph,
                    1 - config["importance_clip_ratio"],
                    1 + config["importance_clip_ratio"],
                )
            )
            self.br_loss = (
                self.pi_loss * importance_ratio * self.config["pi_loss_coeff"]
                + self.mean_vf_loss
                - self.config["entropy_coeff"] * self.approx_ent
                + self.config["kl_coeff"] * self.approx_kl
            )

            self.clipped = tf.logical_or(
                self.ratio > (1 + config["clip_ratio"]),
                self.ratio < (1 - config["clip_ratio"]),
            )
            self.clipfrac = tf.reduce_mean(tf.cast(self.clipped, tf.float32))

            # Optimizers
            self.train_pi = MpiAdamOptimizer(learning_rate=config["pi_lr"]).minimize(
                self.loss
            )
            # br optimizer
            self.br_train_pi = MpiAdamOptimizer(learning_rate=config["br_pi_lr"]).minimize(
                self.br_loss
            )
            # --------------------------TMP objectives---------------------------------------------
            self.ratio_tmp = tf.exp(logp_tmp - self.logp_old_ph)  # pi(a|s) / pi_old(a|s)

            self.adv_mean_tmp = tf.reduce_mean(adv_ph)
            self.approx_kl_tmp = tf.reduce_mean(
                self.logp_old_ph - logp_tmp
            )  # a sample estimate for KL-divergence, easy to compute

            self.approx_ent_tmp = tf.reduce_mean(
                -logp_tmp
            )  # a sample estimate for entropy, also easy to compute

            self.ratio_tmp_clipped = tf.clip_by_value(
                self.ratio_tmp, 1 - config["clip_ratio"], 1 + config["clip_ratio"]
            )
            if self.config["algo"] == "ppo":
                self.pi_loss_tmp = -tf.reduce_mean(
                    tf.minimum(self.ratio_tmp * adv_ph, self.ratio_tmp_clipped * adv_ph)
                )
            else:
                self.pi_loss_tmp = -tf.reduce_mean(self.ratio_tmp * adv_ph)

            # self.v_loss_tmp = tf.reduce_mean((ret_ph - self.v_tmp) ** 2)

            vf_loss1_tmp = tf.square(self.v_tmp - ret_ph)
            vf_clipped_tmp = self.val_ph + tf.clip_by_value(
                self.v_tmp - self.val_ph, -config["vf_clip_param"], config["vf_clip_param"]
            )
            vf_loss2_tmp = tf.square(vf_clipped_tmp - ret_ph)
            vf_loss_tmp = tf.maximum(vf_loss1_tmp, vf_loss2_tmp)
            self.mean_vf_loss_tmp = tf.reduce_mean(vf_loss_tmp)

            self.loss_tmp = (
                self.pi_loss_tmp * self.config["pi_loss_coeff"]
                + self.mean_vf_loss_tmp * self.config["v_loss_coeff"]
                + self.config["kl_coeff"] * self.approx_kl_tmp
                + self.config["entropy_coeff"] * self.approx_ent_tmp
            )

            self.clipped_tmp = tf.logical_or(
                self.ratio_tmp > (1 + config["clip_ratio"]),
                self.ratio_tmp < (1 - config["clip_ratio"]),
            )
            self.clipfrac_tmp = tf.reduce_mean(tf.cast(self.clipped_tmp, tf.float32))

            # Optimizers
            self.train_pi_tmp = MpiAdamOptimizer(learning_rate=config["pi_lr"]).minimize(
                self.loss_tmp
            )

            # ------------------ update policy -----------------------
            self.update_ratio_ph = core.placeholder(None)
            self.keep_ratio_ph = core.placeholder(None)
            t_vars = tf.trainable_variables()
            self.pi_vars = [var for var in t_vars if "pi_origin" in var.name]
            self.pi_tmp_vars = [var for var in t_vars if "pi_tmp" in var.name]
            self.v_vars = [var for var in t_vars if "v_origin" in var.name]
            self.v_tmp_vars = [var for var in t_vars if "v_tmp" in var.name]

    def update_policy(self, sess, ratio):
        # update the original policies and critics with the update ratio
        # and copy original policies and critic to tmp
        for i in range(len(self.pi_vars)):
            origin_pi_var = tf.math.multiply(self.pi_vars[i], ratio[0])
            tmp_pi_var = tf.math.multiply(self.pi_tmp_vars[i], ratio[1])
            update_pi_var = tf.math.add(origin_pi_var, tmp_pi_var)
            sess.run(tf.assign(self.pi_vars[i], update_pi_var))
            # sess.run(tf.assign(self.pi_tmp_vars[i], self.pi_vars[i]))
        for i in range(len(self.v_vars)):
            origin_v_var = tf.math.multiply(self.v_vars[i], ratio[0])
            tmp_v_var = tf.math.multiply(self.v_tmp_vars[i], ratio[1])
            update_v_var = tf.math.add(origin_v_var, tmp_v_var)
            sess.run(tf.assign(self.v_vars[i], update_v_var))
            # sess.run(tf.assign(self.v_tmp_vars[i], self.v_vars[i]))

    def update_original_policy(self, sess, data):

        inputs = {k: v for k, v in zip(self.all_phs, data)}
        # Training
        for i in range(self.config["train_pi_iters"]):
            mini_batch = sample_batch(data, self.config["batch_size"])
            mini_batch_input = {k: v for k, v in zip(self.all_phs, mini_batch)}
            _, kl = sess.run(
                [self.train_pi, self.approx_kl], feed_dict=mini_batch_input
            )
            kl = mpi_avg(kl)
            if kl > 1.5 * self.config["target_kl"]:
                print("Early stopping at step %d due to reaching max kl." % i)
                break
        # self.logger.store(StopIter=i)

        # Log changes from update
        total_l_new, pi_l_new, v_l_new, kl, ratio, ent, cf = sess.run(
            [
                self.loss,
                self.pi_loss,
                self.mean_vf_loss,
                self.approx_kl,
                self.ratio,
                self.approx_ent,
                self.clipfrac,
            ],
            feed_dict=inputs,
        )

        return (
            total_l_new,
            pi_l_new,
            v_l_new,
            kl,
            ent,
            ratio,
            cf,
        )

    def update_tmp(self, sess, data):
        inputs = {k: v for k, v in zip(self.all_phs, data)}
        # Training
        for i in range(self.config["train_pi_iters"]):
            _, kl, entropy = sess.run(
                [self.train_pi_tmp, self.approx_kl_tmp, self.approx_ent_tmp],
                feed_dict=inputs,
            )
            kl = mpi_avg(kl)
            if kl > 1.5 * self.config["target_kl"]:
                print("Early stopping at step %d due to reaching max kl." % i)
                break
            if entropy < 1.5:
                self.config["entropy_coeff"] = 0.02
            if entropy < 1.3:
                self.config["entropy_coeff"] = 0.04
            if entropy < 1.1:
                self.config["entropy_coeff"] = 0.08
        # self.logger.store(StopIterTmp=i)

        # Log changes from update
        total_l_new, pi_l_new, v_l_new, kl, ratio_tmp, cf = sess.run(
            [
                self.loss_tmp,
                self.pi_loss_tmp,
                self.mean_vf_loss_tmp,
                self.approx_kl_tmp,
                self.ratio_tmp,
                self.clipfrac_tmp,
            ],
            feed_dict=inputs,
        )
        return (
            total_l_new,
            pi_l_new,
            v_l_new,
            kl,
            entropy,
            ratio_tmp,
            cf,
        )

    def get_logp_ratio(self, sess, obs, act, logp, policy):
        if policy == "original":
            return 1.0
        else:
            return sess.run(
                self.ratio_tmp,
                feed_dict={self.x_ph: obs, self.a_ph: act, self.logp_old_ph: logp},
            )

    def get_logp(self, sess, obs, act, logp):

        return sess.run(
            self.ratio,
            feed_dict={self.x_ph: obs, self.a_ph: act, self.logp_old_ph: logp},
        )

    def br_update(self, sess, data, ratio):
        # update the agent policy with a importance ratio

        inputs = {k: v for k, v in zip(self.all_phs, data)}
        inputs[self.importance_ratio_ph] = ratio
        for i in range(self.config["br_iter"]):
            _, pi_loss = sess.run([self.br_train_pi, self.br_loss], feed_dict=inputs)
        return pi_loss

    def get_action(self, sess, obs, policy):
        if self.is_cnn or self.is_rnn:
            obs = np.array([obs])
        else:
            obs = np.reshape(obs, (1, -1))
        if policy == "tmp":
            return sess.run(self.get_tmp_action_ops, feed_dict={self.x_ph: obs})

        return sess.run(self.get_action_ops, feed_dict={self.x_ph: obs})

    def get_value(self, sess, obs, policy):
        if self.is_cnn or self.is_rnn:
            if len(np.array(obs).shape) == 3:
                obs = np.array([obs])
            else:
                obs = np.array(obs)
        else:
            if len(np.array(obs).shape) == 1:
                obs = np.reshape(obs, (1, -1))
            else:
                obs = np.array(obs)
        if policy == "tmp":
            return sess.run(self.v_tmp, feed_dict={self.x_ph: obs})

        return sess.run(self.v, feed_dict={self.x_ph: obs})


def sample_batch(inputs, bs):
    indexes = np.random.choice(len(inputs), bs)
    return [i[indexes] for i in inputs]
