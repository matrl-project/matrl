import gym 
import ma_gym
# from utils.soccer_wrapper import DMSoccer

from utils.atari_wrapper import AtariEnv
from multiagent_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

def init_env(config): 
    env_info = None
    if config["game_type"] == "grid":
        env = gym.make(config["env"])
        n_agents = env.n_agents
    elif config["game_type"] == "soccer":
        env = DMSoccer(
            team_size=config["soccer"]["team_size"],
            time_limit=config["soccer"]["time_limit"],
        )
        n_agents = env.n_agents
    elif config["game_type"] == "mujoco":
        env_args = {"scenario": config["env"], "episode_limit": 1000}
        if config["env"] == "Hopper-v2":
            env_args["agent_conf"] = "3x1"
            # env_args["agent_obsk"]=1
        if config["env"] == "HalfCheetah-v2":
            env_args["agent_conf"] = "2x3"
            # env_args["agent_obsk"]=1
        if config["env"] == "Ant-v2":
            env_args["agent_conf"] = "2x4"
        if config["env"] == "Humanoid-v2":
            env_args["agent_conf"] = "9|8"
        if config["env"] == "HalfCheetah-v2":
            env_args["agent_conf"] = "2x3"
        if config["env"] == "Reacher-v2":
            env_args["agent_conf"] = "2x1"
        if config["env"] == "Swimmer-v2":
            env_args["agent_conf"] = "2x1"
        env = MujocoMulti(env_args=env_args)
        env_info = env.get_env_info()
        n_agents = env_info["n_agents"]
    elif config["game_type"] == "atari":
        env = AtariEnv(config["env"])
        n_agents = env.n_agents
    else:
        raise ValueError("game type not defined")

    return env, n_agents, env_info