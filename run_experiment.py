from spinup.utils.mpi_tools import mpi_fork, num_procs
from spinup.utils.run_utils import setup_logger_kwargs

from multiagent_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from configs.config import *
from datetime import datetime
import yaml

import os

import tensorflow as tf

from utils.init_env import init_env

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument('--meta', dest='meta', action='store_true')
    parser.add_argument('--no-meta', dest='meta', action='store_false')
    parser.set_defaults(meta=True)
    parser.add_argument('--br', dest='br', action='store_true')
    parser.add_argument('--no-br', dest='br', action='store_false')
    parser.set_defaults(br=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="GPU")

    args = parser.parse_args()
    device = args.device
    if device == "CPU" or device == "GPU":
        device = f"/{args.device}:0"
    elif device not in ["/GPU:1", "/GPU:2", "/GPU:3"]:
        raise ValueError
    a_yaml_file = open(args.config_file)
    config = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    config = update_config(config)

    mpi_fork(config["cpu"])  # run parallel code with mpi
    config["seed"] = args.seed
    now = datetime.now()
    config["output_dir"] = "{}/{}/{}-{}-{}-{}-{}-{}".format(
        config["output_dir"],
        config["env"],
        config["trainer"],
        config["exp_name"],
        config["algo"],
        config["network"],
        config["share_policy"],
        config["seed"],
    )

    if config["trainer"] == "meta":
        config["output_dir"] += "-{}".format(config["br"])
    config["output_dir"] += "-{}".format(now.strftime("%Y%m%d%H:%M:%S"))
    config["save_path"] = os.path.join(config["output_dir"], config["save_path"])
    
    config["meta"] = args.meta 
    config["br"] = args.br
    env, n_agents, env_info = init_env(config)

    print("output folder is: ", config["output_dir"])
    print("env info: ", env.observation_space, env.action_space, env.n_agents)
    with tf.device(device):
        trainer = TRAINERs[config["trainer"]](env, env.n_agents, config)
        trainer.train()
    
    # create time logging 

    later_time = datetime.now()
    difference = later_time - now
    print(later_time.strftime("%Y%m%d%H:%M:%S"))
    
    with open(config["output_dir"] + "/" + 'time.txt', 'a') as a_writer:
        a_writer.write('Total time in seconds: {}'.format(difference.total_seconds() ))
    
