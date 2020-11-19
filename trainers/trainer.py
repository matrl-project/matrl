import yaml
import pickle
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import sync_all_params
from spinup.utils.mpi_tools import mpi_fork, num_procs
import os
from utils.logger import Logger
from utils.buffer import new_buffer
import tensorflow as tf
from agents.pg_general import PGAgent
import gym


class Trainer:
    def __init__(self, env, num_agents, config):
        self.num_agent = num_agents
        self.config = config
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.is_rnn = config["network"] == "rnn"
        self.is_cnn = config["network"] == "cnn"
        config["num_agent"] = num_agents
        if isinstance(self.observation_space, gym.spaces.Box):

            if config["share_policy"]:
                pg_agent = PGAgent(
                    "shared",
                    self.observation_space,
                    gym.spaces.Discrete(self.action_space.shape[0] / self.num_agent),
                    config,
                )
                self.agents = [pg_agent for i in range(num_agents)]
            else:
                self.agents = [
                    PGAgent(
                        i,
                        self.observation_space,
                        gym.spaces.Discrete(
                            self.action_space.shape[0] / self.num_agent
                        ),
                        config,
                    )
                    for i in range(num_agents)
                ]
        else:
            if config["share_policy"]:
                pg_agent = PGAgent(
                    "shared", self.observation_space[0], self.action_space[0], config,
                )
                self.agents = [pg_agent for i in range(num_agents)]
            else:
                self.agents = [
                    PGAgent(i, self.observation_space[i], self.action_space[i], config)
                    for i in range(num_agents)
                ]
        self.local_steps_per_epoch = int(config["steps_per_epoch"] / num_procs())
        if type(self.action_space[0]) is gym.spaces.Discrete:
            action_dim = 1
        else:
            action_dim = self.action_space[0].shape[0]
        if type(self.observation_space[0]) is gym.spaces.Discrete:
            obs_dim = self.observation_space[0].n
        else:
            if self.is_cnn:
                obs_dim = self.observation_space[0].shape
            else:
                obs_dim = self.observation_space[0].shape[0]
        self.buf = new_buffer(
            num_agents,
            obs_dim,
            action_dim,
            size=self.local_steps_per_epoch,
            type=config["algo"],
            is_rnn=self.is_rnn,
            is_cnn=self.is_cnn,
            rnn_length=config["rnn_length"],
        )

        # init logger
        
        self.logger = Logger(self.config)

        # init session and sync params
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(sync_all_params())

        tf.io.write_graph(
            graph_or_graph_def=self.sess.graph_def,
            logdir=os.path.join(self.config["output_dir"]),
            name="model",
        )
        # save model
        self.savers =[] 
        with tf.device('/cpu:0'):
            for i in range(num_agents):
                vars_list = tf.compat.v1.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope="Agent_{}".format(i)
                )
                self.savers.append(tf.train.Saver(vars_list, max_to_keep=100))

        if self.config["save_path"] is None:
            self.save_path = "/tmp/agents.pickle"
        else:
            self.save_path = self.config["save_path"]
            self.save_path += 's/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # load model
        if self.config["load_model"]:
            self.restore(self.config["restore_path"], [0, 1])

    def train(self):
        pass

    def save(self, step):
        for i in range(self.num_agent):
            self.savers[i].save(self.sess, os.path.join(self.save_path, "model_{}.cpt".format(i)) , global_step=step)

    def restore(self, restore_paths, agent_ids):
        for i in range(self.num_agent):
            if restore_paths[i] != "":
                self.savers[agent_ids[i]].restore(self.sess, restore_paths[i])
