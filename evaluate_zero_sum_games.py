import tensorflow as tf
import yaml
import gym
import ma_gym
import os
import numpy as np
from utils.init_env import init_env
from configs.config import *
import glob


def calculate_nash_conv(env, config, arg, devices):
    n_agents = env.n_agents
    checkpoint_path = args.checkpoint_path
    with tf.device(device):
        current_trainer = TRAINERs[config["trainer"]](env, n_agents, config)
        current_trainer.restore(checkpoint_path)

        # to get the current policy evaluate value
        current_values = []
        current_trainer.buf.clear()
        current_trainer.run_one_epoch(["original"] * n_agents)
        current_data = current_trainer.buf.get()
        # n_agents x shape of state
        obs = current_data[0]

        for j in range(n_agents):
            values = current_trainer.agents[j].get_value(
                current_trainer.sess, np.array(obs[:, j]), "original"
            )
            # n_agents x n_states
            current_values.append(values)

        tf.compat.v1.reset_default_graph()
        br_values = []

        br_trainer = TRAINERs[config["trainer"]](env, n_agents, config)
        br_trainer.restore(checkpoint_path)
        br_trainer.rollouts(1, ["original"] * n_agents)
        br_trainer.update_tmp_policies()

        for i in range(args.num_games_eval):
            br_trainer.buf.clear()
            br_trainer.run_one_epoch(["tmp"] * n_agents)
            br_data = br_trainer.buf.get()
            for j in range(n_agents):
                values = br_trainer.agents[j].get_value(
                    br_trainer.sess, np.array(obs[:, j]), "tmp"
                )
                br_values.append(values)
                # br_values[agent].append(br_data[3][-1, agent])
            # br_ret = [i[-1] for i in br_data[3]]
            # br_values.append(br_ret)

            # current_values [n_agents x n_states]
            nash_conv = 0
            for i in range(n_agents):
                nash_conv += sum(
                    np.array(br_values[i][:]) - np.array(current_values[i][:])
                )

            # nash_conv = sum([b - c] for b,c in zip(br_values ,current_values))

            return nash_conv


def calculate_score(env, sess, agents):
    """
    rollout with two policies, find the score of using policy1 against policy 2 
    """
    n_agents = env.n_agents
    obs_n, rew_n, done_n, ep_ret_n, ep_len = (
        env.reset(),
        [0 for i in range(n_agents)],
        False,
        [0 for i in range(n_agents)],
        0,
    )
    done = False
    while not done:
        act_n = []
        for i in range(n_agents):
            action, _, _ = agents[i].get_action(sess, obs_n[i], "original")
            act_n.append(action[0])
        obs_n, rew_n, done_n, _ = env.step(act_n)
        ep_ret_n = [n + r for n, r in zip(ep_ret_n, rew_n)]
        ep_len += 1
        if np.array(done_n).any():
            done = True

    return ep_ret_n


def calculate_population_performance(env, config, args, device):
    matrix_size = 2
    matrix = np.zeros((matrix_size, matrix_size))
    # load saved policies
    n_agents = env.n_agents
    checkpoints = []
    print(args.checkpoint_path)
    for i in range(n_agents):
        model_checkpoints =[] 
        for j in range(matrix_size):
            model_checkpoints.append(args.checkpoint_path[i] + "model_{}.cpt-{}".format(i, j))
        checkpoints.append(model_checkpoints)
    # model_checkpoints_2 = glob.glob(os.path.join(args.checkpoint_path_2, "*.pt"))

    trainer = TRAINERs[config["trainer"]](env, n_agents, config)
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i != j :
                tf.compat.v1.reset_default_graph()
                trainer.restore([checkpoints[0][i], checkpoints[1][j]])
                score = calculate_score(env, trainer.sess, trainer.agents)
                matrix[i][j] = score[0]
    return matrix


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--checkpoint_path", nargs='+', default="")
    parser.add_argument("--evaluate_method", type=str, default="nash_conv")
    parser.add_argument("--num_games_eval", type=int, default=10)
    parser.add_argument("--output_folder", type=str, default="evaluate_logs")
    parser.add_argument("--device", type=str, default="CPU")
    args = parser.parse_args()

    a_yaml_file = open(args.config_file)
    config = yaml.load(a_yaml_file, Loader=yaml.FullLoader)

    device = args.device
    if device == "CPU" or device == "GPU":
        device = f"/{args.device}:0"
    elif device not in ["/GPU:1", "/GPU:2", "/GPU:3"]:
        raise ValueError

    env, n_agents, env_info = init_env(config)

    print("output folder is: ", config["output_dir"])
    print("env info: ", env.observation_space, env.action_space, env.n_agents)

    if args.evaluate_method == "nash_conv":
        nash_conv = calculate_nash_conv(env, config, args, device)
        print("evaluate score -- nash conv is: ", nash_conv)
    elif args.evaluate_method == "population_performance":
        matrix = calculate_population_performance(env, config, args, device)
        print(matrix)
