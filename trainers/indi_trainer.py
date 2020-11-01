from trainers.trainer import Trainer
import numpy as np


class IndiTrainer(Trainer):
    def __init__(self, env, num_agents, config):
        super().__init__(env, num_agents, config)
        self.keys = [
            "total_loss",
            "policy_loss",
            "value_loss",
            "kl",
            "entropy",
            "policy_ratio",
            "clip_ratio",
            "adv",
            "value",
            "episode_return",
            "episode_length",
        ]

    def run_one_epoch(self, policies):
        """
        this function execute the multi agent game for one episode
        """
        obs_n, rew_n, done_n, ep_ret_n, ep_len = (
            self.env.reset(),
            [0 for i in range(self.num_agent)],
            False,
            [0 for i in range(self.num_agent)],
            0,
        )
        for t in range(self.local_steps_per_epoch):

            if self.is_rnn:
                self.buf.store_pure_obs(obs_n)
                obs_n = self.buf.get_rnn_obs(obs_n)
            act_n, val_t_n, logp_t_n = self.get_actions(obs_n, policies)
            obs_n_next, rew_n, done_n, _ = self.env.step(act_n)
            if type(rew_n) not in [list, np.ndarray]: 
                rew_n = [rew_n] * self.num_agent

            ep_ret_n = [n + r for n, r in zip(ep_ret_n, rew_n)]
            ep_len += 1
            # collect data
            act_n = np.array(act_n).reshape((len(obs_n), -1))
            self.buf.store(obs_n,np.array(act_n), rew_n, val_t_n, logp_t_n)
            self.logger.store({"value": val_t_n}, [i for i in range(self.num_agent)])

            obs_n = obs_n_next

            terminal = np.all(done_n)  or (ep_len == self.config["max_ep_len"])
            if terminal or (t == self.local_steps_per_epoch - 1):
                if not (terminal):
                    if self.is_rnn:
                        obs_n = self.buf.get_rnn_obs(obs_n)
                    print("Warning: trajectory cut off by epoch at %d steps." % ep_len)
                    last_val_n = (
                        [0 for i in range(self.num_agent)]
                        if np.all(done_n)
                        else self.get_value(obs_n, policies)
                    )
                    self.buf.finish_path(last_val_n)
                else :
                    for i in range(len(self.agents)):
                        self.logger.store(
                            {"episode_return": ep_ret_n, "episode_length": [ep_len] * self.num_agent},
                            [i for i in range(self.num_agent)],
                        )
                    last_val_n = (
                        [0 for i in range(self.num_agent)]
                        if np.all(done_n)
                        else self.get_value(obs_n, policies)
                    )
                    self.buf.finish_path(last_val_n)
                obs_n, rew_n, done_n, ep_ret_n, ep_len = (
                    self.env.reset(),
                    [0 for i in range(self.num_agent)],
                    False,
                    [0 for i in range(self.num_agent)],
                    0,
                )

    def rollouts(self, epochs, policies):
        """
        this function will execute the game for multiple episodes to collect the data save to buffer
        """
        for _ in range(epochs):
            self.run_one_epoch(policies)

    def get_actions(self, obs_n, policies):
        actions = []
        values = []
        logps = []
        for i in range(len(policies)):
            action, value, logp = self.agents[i].get_action(
                self.sess, obs_n[i], policies[i]
            )
            actions.append(action[0])
            values.append(value[0])
            logps.append(logp[0])
        return actions, values, logps

    def get_value(self, obs_n, policies):
        # obs_n [batch, num_agent, space_shape]
        values = []
        for i in range(len(policies)):
            value = self.agents[i].get_value(self.sess, obs_n[i], policies[i])
            values.append(value[0])
        return values

    def update_policies(self):
        data = self.buf.get()
        for i in range(self.num_agent):
            agent_data = [item[:, i] for item in data]
            total_loss, p_loss, v_loss, kl, ent, ratio, clip_ratio = self.agents[
                i
            ].update_original_policy(self.sess, agent_data)

            self.logger.store(
                {   
                    "total_loss": total_loss, 
                    "policy_loss": p_loss,
                    "value_loss": v_loss,
                    "kl": kl,
                    "entropy": ent,
                    "policy_ratio": np.mean(ratio),
                    "clip_ratio": clip_ratio,
                    "adv": np.mean(data[i][2]),
                },
                agent=[i],
            )
        self.buf.clear()

    def train(self):
        for epoch in range(self.config["epochs"]):
            self.rollouts(self.config["tmp_rollout_epochs"], ["original"] * self.num_agent)
            self.update_policies()

            self.logger.dump(
                self.keys,
                agents=[i for i in range(len(self.agents))],
                step=epoch,
                mean_only=True,
            )

            if epoch % self.config["save_frequency"] == 0 : 
                print("--- the model has been saved ---")
                self.save()
