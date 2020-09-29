from pettingzoo.atari import (
    boxing_v0,
    pong_foozpong_v0,
    pong_volleyball_v0,
    pong_classic_v0,
)
from gym.spaces import Box, Discrete
import gym
import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np 


class AtariEnv(gym.Env):
    def __init__(self, env_name: str):
        player_names = ["first", "second", "third", "fourth"]
        if env_name == "pong":
            self.env = pong_classic_v0.env()
            self.n_agents = 2
            self.action_space = [Discrete(18)] * self.n_agents
        if env_name == "boxing":
            self.env = boxing_v0.env()
            self.n_agents = 2
            self.action_space = [Discrete(18)] * self.n_agents
        if env_name == "foozpong":
            self.n_agents = 4
            self.env = pong_foozpong_v0.env(num_players=self.n_agents)
            self.action_space = [Discrete(18)] * self.n_agents
        if env_name == "pong_volleyball_v0":
            self.n_agents = 4
            self.env = pong_volleyball_v0.env(num_players=self.n_agents)
            self.action_space = [Discrete(18)] * self.n_agents

        self._width = 84
        self._height = 84
        self.env.reset()
        self.agents = [f"{player_names[n]}_0" for n in range(self.n_agents)]
        self.observation_space = [Box(low=0, high=1, shape=(84,84,3))] * self.n_agents
        self.reverse = False

    def step(self, acts):

        obs, rewards, dones, infos = [], [], [], []
        indexes = [n for n in range(self.n_agents)]
        if self.reverse:
            indexes = reversed(indexes)
        for i in indexes:
            observation = self.env.step(acts[i])
            reward, done, info = self.env.last()
            obs.append(self.process_obs(observation))
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return obs, rewards, dones, infos

    def process_obs(self, obs):
        frame = cv2.resize(
            obs, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        return np.array(frame).astype(np.float32) / 255.0

    def reset(self):
        obs = self.env.reset()
        obs = self.process_obs(obs)
        self.reverse = not self.reverse
        return [obs for i in range(self.n_agents)]
