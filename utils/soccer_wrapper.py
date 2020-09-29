import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from multiagent_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from gym.spaces import Box
import sys


class DMSoccer(MujocoMulti):
    def __init__(self, batch_size=None, **kwargs):
        # super().__init__(batch_size, **kwargs)
        self.base_env = dm_soccer.load(kwargs["team_size"], kwargs["time_limit"])
        act_spec = self.base_env.action_spec()
        self.n_agents = len(act_spec)
        self.observation_space = [
            Box(low=-10000, high=10000, shape=(154,))
        ] * self.n_agents
        self.action_space = [Box(low=-1, high=1, shape=(3,))] * self.n_agents

    def parse_obs(self, observation):
        obs = [[]] * self.n_agents
        for i in range(self.n_agents):
            for _, v in observation[i].items():
                v = v.flatten()
                for j in range(len(v)):
                    obs[i].append(float(v[j]))
        return obs

    def get_reward(self, observation, origin_reward):
        """if goal, return the original reward, else, return the shaped dense reward
        """
        rewards = [0.0] * self.n_agents
        for i in range(self.n_agents):
            if not origin_reward or origin_reward[i] == 0.0:
                home_score = observation[i]["stats_home_score"]
                away_score = observation[i]["stats_away_score"]
                velocity_to_ball = observation[i]["stats_vel_to_ball"]
                velocity_to_goal = observation[i]["stats_vel_ball_to_goal"]
                rewards[i] = (
                    home_score
                    - away_score
                    + 0.001 * velocity_to_ball
                    + 0.002 * velocity_to_goal
                )
            else:
                rewards[i] = origin_reward[i]
        return rewards

    def step(self, actions):
        try:
            timestep = self.base_env.step(actions)
        except:
            obs = self.parse_obs(self.base_env.reset().observation)
            return obs, [0] * self.n_agents, True, None
        # concatenate all the values in the observation dict as the observation for each agent
        obs = self.parse_obs(timestep.observation)
        done = timestep.discount == 0

        reward = self.get_reward(timestep.observation, timestep.reward)
        return obs, reward, done, None

    def reset(self, **kwargs):
        timestep = self.base_env.reset()
        obs = self.parse_obs(timestep.observation)
        return obs

    def render(self, **kwargs):
        self.base_env.render()
