import numpy as np
from spinup.utils.mpi_tools import (
    mpi_fork,
    mpi_avg,
    proc_id,
    mpi_statistics_scalar,
    num_procs,
)
import utils.core as core
import collections


def new_buffer(num_agent, obs_dim, act_dim, size, type, **kwargs):
    return MetaPPOBuffer(num_agent, obs_dim, act_dim, size, **kwargs)


class MetaPPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(
        self, num_agent, obs_dim, act_dim, size, gamma=0.99, lam=0.95, **kwargs
    ):
        self.num_agent = num_agent
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size = size
        self.is_rnn = kwargs["is_rnn"]
        self.is_cnn = kwargs["is_cnn"]
        self.rnn_length = kwargs["rnn_length"]
        self.gamma = gamma
        self.lam = lam
        self.clear()

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """

        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def store_pure_obs(self, obs):
        # prev_obs = [num_agent, rnn_length, obs_dim]
        prev_obs = np.array(self.pure_obs_buf[-1])
        # obs_n = num_agent, obs_dim,
        prev_obs[:, -1, :] = np.array(obs)
        self.pure_obs_buf[self.ptr] = prev_obs

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], np.array([last_val]), axis=0)
        vals = np.append(self.val_buf[path_slice], np.array([last_val]), axis=0)
        # the next two lines implement GAE-Lambda advantage calculation for all agents
        for i in range(self.num_agent):
            # rews , vals, adv_buf : [step, num_agent]
            deltas = rews[:-1, i] + self.gamma * vals[1:, i] - vals[:-1, i]
            self.adv_buf[path_slice, i] = core.discount_cumsum(
                deltas, self.gamma * self.lam
            )
            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice, i] = core.discount_cumsum(rews[:, i], self.gamma)[
                :-1
            ]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self.statistics_scalar(self.adv_buf)

        for i in range(self.num_agent):
            if adv_std[i] == 0:
                adv_std[i] = 1
            assert adv_std[i] != 0, "adv_std is 0 not valid"
            self.adv_buf[:, i] = (self.adv_buf[:, i] - adv_mean[i]) / adv_std[i]
        return [
            self.obs_buf,
            self.act_buf,
            self.adv_buf,
            self.ret_buf,
            self.logp_buf,
            self.val_buf,
        ]

    def get_tmp(self):
        # assert self.ptr == self.max_size  # buffer has to be full before you can get
        # self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self.statistics_scalar(self.adv_tmp)
        # if adv_std == 0:
        #     adv_std = 1
        for i in range(self.num_agent):
            if adv_std[i] == 0:
                adv_std[i] = 1
            assert adv_std[i] != 0, "adv_std is 0 not valid"
            self.adv_tmp[:, i] = (self.adv_tmp[:, i] - adv_mean[i]) / adv_std[i]
        return [
            self.obs_tmp,
            self.act_tmp,
            self.adv_tmp,
            self.ret_tmp,
            self.logp_tmp,
            self.val_tmp,
        ]

    def keep_tmp_buf(self):
        self.obs_tmp = np.copy(self.obs_buf)
        self.act_tmp = np.copy(self.act_buf)
        self.adv_tmp = np.copy(self.adv_buf)
        self.rew_tmp = np.copy(self.rew_buf)
        self.ret_tmp = np.copy(self.ret_buf)
        self.val_tmp = np.copy(self.val_buf)
        self.logp_tmp = np.copy(self.logp_buf)

    def clear(self):
        if self.is_cnn:
            self.obs_buf = np.zeros(
                shape=(
                    self.size,
                    self.num_agent,
                    self.obs_dim[0],
                    self.obs_dim[1],
                    self.obs_dim[2],
                )
            )
        elif self.is_rnn:
            self.pure_obs_buf = np.zeros(
                (self.size, self.num_agent, self.rnn_length, self.obs_dim),
                dtype=np.float32,
            )
            self.obs_buf = np.zeros(
                (
                    self.size,
                    self.num_agent,
                    self.rnn_length,
                    self.obs_dim + self.num_agent * self.act_dim,
                ),
                dtype=np.float32,
            )
        else:
            self.obs_buf = np.zeros(
                (self.size, self.num_agent, self.obs_dim), dtype=np.float32
            )
        self.act_buf = np.zeros(
            (self.size, self.num_agent, self.act_dim), dtype=np.float32
        )

        self.adv_buf = np.zeros((self.size, self.num_agent), dtype=np.float32)
        self.rew_buf = np.zeros((self.size, self.num_agent), dtype=np.float32)
        self.ret_buf = np.zeros((self.size, self.num_agent), dtype=np.float32)
        self.val_buf = np.zeros((self.size, self.num_agent), dtype=np.float32)
        self.logp_buf = np.zeros((self.size, self.num_agent), dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.size

    def get_rnn_obs(self, obs_n):
        # prev_obs = [num_agent, rnn_length, obs_dim]
        prev_obs = np.array(self.pure_obs_buf[-1])
        # obs_n = num_agent, obs_dim,
        prev_obs[:, -1, :] = np.array(obs_n)
        # prev acts = length x num_agent x obs_dim
        prev_act = self.act_buf[-self.rnn_length :]
        # extend acts to num_agent, rnn_length,  num_action
        prev_act = np.expand_dims(prev_act, axis=-1)
        prev_act = np.transpose(prev_act, (1, 0, 2))
        prev_act = list(prev_act)
        for i in range(self.num_agent):
            prev_act[i] = list(prev_act[i])
            for j in range(self.rnn_length):
                index = prev_act[i][j]
                hist = []
                if i == 0:
                    for k in range(self.num_agent):
                        cur_index = prev_act[k][j]
                        hist.append(
                            [1 if cur_index == g else 0 for g in range(self.act_dim)]
                        )
                    hist = np.array(hist)
                    hist = hist.flatten()
                    prev_act[i][j] = hist
                else:
                    prev_act[i][j] = prev_act[0][j]
        prev_act = np.array(prev_act)
        prev_act = np.array(prev_act)
        obs = np.concatenate([prev_obs, prev_act], axis=2)
        return obs

    def statistics_scalar(self, data):
        means, stds = [], []
        for i in range(self.num_agent):
            means.append(np.mean(data[:, i]))
            stds.append(np.std(data[:, i]))
        return means, stds
