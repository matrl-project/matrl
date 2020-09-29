from trainers.indi_trainer import IndiTrainer
import numpy as np
import tensorflow as tf
import gym
import time
import utils.core as core
from spinup.utils.mpi_tf import sync_all_params
from spinup.utils.mpi_tools import mpi_fork, num_procs
from spinup.utils.run_utils import setup_logger_kwargs
import yaml
import os
from datetime import datetime
from nash_solver3.game import Game
from matrix_games.util import check_all_positive


class MetaTrainer(IndiTrainer):
    def __init__(self, env, num_agents, config):
        super().__init__(env, num_agents, config)
        # init policy and tmp policy with same set of weights
        for agent in self.agents:
            agent.update_policy(self.sess, [i for i in range(self.num_agent)])
        self.keys.append("nash_pair")
        if config["br"]:
            self.keys.append("br_policy_loss")
            self.keys.append("br_importance_rate")

    def update_tmp_policies(self):
        """
        This function updates the tmp policy based on the rollouts data
        """
        data = self.buf.get()

        for i in range(self.num_agent):
            agent_data = [item[:, i] for item in data]
            agent = self.agents[i]
            total_loss, p_loss, v_loss, kl, ent, ratio, clip_ratio = agent.update_tmp(
                self.sess, agent_data
            )
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
        self.buf.keep_tmp_buf()
        self.buf.clear()

    def update_policies_with_nash(self):
        meta_policies_weight = self.calculate_meta_game_policies()
        for i in range(len(self.agents)):
            self.agents[i].update_policy(self.sess, meta_policies_weight[i])

    def calculate_meta_game_policies(self):
        """TODO: use the trajectories calculated to get the meta game and then get the next policy based on meta game result 
        1. rollouts with different combinations of policies 

        """
        if self.config["meta"]:
            matrix = self.compute_payoff_matrix()
            game = {
                "name": "generated game",
                "players": ["Player {}".format(i) for i in range(self.num_agent)],
                "num_players": self.num_agent,
                "shape": [2 for i in range(self.num_agent)],
                "sum_shape": sum([2 for i in range(self.num_agent)]),
                "array": matrix,
            }

            g = Game(game)
            result = g.findEquilibria(self.config["nash_method"])
            if result == None:
                result = [[1, 0] for i in range(self.num_agent)]
            else:
                result[0].normalize()
                result = result[0]._list
                if not check_all_positive(result):
                    result = [[1, 0] for i in range(self.num_agent)]
        else:
            result = [[0, 1] for i in range(self.num_agent)]
        self.logger.store(
            {"nash_pair": [i[0] for i in result]},
            agent=[i for i in range(self.num_agent)],
        )
        return result

    def calculate_policy_adv(self, main_agent, policies):
        if self.config["meta_rollouts"]:
            # do rollouts to get the advantage
            self.rollouts(self.config["meta_rollout_epochs"], policies)
            # calculate the policy advantage
            data = self.buf.get_tmp()
            adv_data = data[2][main_agent]
            # clear the buffer after each rollouts
            self.buf.clear()
            return np.mean(adv_data)
        else:
            # use the policy ratio and the rollout experiments to calculate the advantage
            data = self.buf.get_tmp()
            ratio = 1
            for i in range(self.num_agent):
                ratio *= self.agents[i].get_logp_ratio(
                    self.sess,
                    data[0][:, i, :],
                    data[1][:, i],
                    data[4][:, i],
                    policies[i],
                )
            adv_data = data[2][main_agent]
            return np.mean(adv_data) * np.mean(ratio)

    def two_player_matrix_constructor(self):
        matrix = []
        for i in range(self.num_agent):
            adv_origin_tmp = self.calculate_policy_adv(i, ["original", "tmp"])
            adv_tmp_origin = self.calculate_policy_adv(i, ["tmp", "original"])
            adv_tmp_tmp = self.calculate_policy_adv(i, ["tmp", "tmp"])
            A = np.array([[0, adv_origin_tmp], [adv_tmp_origin, adv_tmp_tmp],])
            matrix.append(A)
        return matrix

    def three_player_matrix_constructor(self):
        matrix = []
        for i in range(self.num_agent):
            adv_origin_origin_tmp = self.calculate_policy_adv(
                i, ["original", "original", "tmp"]
            )
            adv_origin_tmp_origin = self.calculate_policy_adv(
                i, ["original", "tmp", "original"]
            )
            adv_origin_tmp_tmp = self.calculate_policy_adv(
                i, ["original", "tmp", "tmp"]
            )
            adv_tmp_origin_origin = self.calculate_policy_adv(
                i, ["tmp", "original", "origin"]
            )
            adv_tmp_origin_tmp = self.calculate_policy_adv(
                i, ["tmp", "original", "tmp"]
            )
            adv_tmp_tmp_origin = self.calculate_policy_adv(
                i, ["tmp", "tmp", "original"]
            )
            adv_tmp_tmp_tmp = self.calculate_policy_adv(i, ["tmp", "tmp", "tmp"])

            A = np.array(
                [
                    [
                        [0, adv_origin_origin_tmp],
                        [adv_origin_tmp_origin, adv_origin_tmp_tmp],
                    ],
                    [
                        [adv_tmp_origin_origin, adv_tmp_origin_tmp],
                        [adv_tmp_tmp_origin, adv_tmp_tmp_tmp],
                    ],
                ]
            )
            matrix.append(A)
        return matrix

    def four_player_matrix_constructor(self):
        matrix = []
        for i in range(self.num_agent):
            adv_origin_origin_origin_tmp = self.calculate_policy_adv(
                i, ["original", "original", "original", "tmp"]
            )
            adv_origin_origin_tmp_origin = self.calculate_policy_adv(
                i, ["original", "original", "tmp", "origin"]
            )
            adv_origin_origin_tmp_tmp = self.calculate_policy_adv(
                i, ["original", "original", "tmp", "tmp"]
            )
            adv_origin_tmp_origin_origin = self.calculate_policy_adv(
                i, ["original", "tmp", "original", "origin"]
            )
            adv_origin_tmp_origin_tmp = self.calculate_policy_adv(
                i, ["original", "tmp", "original", "tmp"]
            )
            adv_origin_tmp_tmp_origin = self.calculate_policy_adv(
                i, ["original", "tmp", "tmp", "origin"]
            )
            adv_origin_tmp_tmp_tmp = self.calculate_policy_adv(
                i, ["original", "tmp", "tmp", "tmp"]
            )

            adv_tmp_origin_origin_origin = self.calculate_policy_adv(
                i, ["tmp", "original", "original", "original"]
            )
            adv_tmp_origin_origin_tmp = self.calculate_policy_adv(
                i, ["tmp", "original", "original", "tmp"]
            )
            adv_tmp_origin_tmp_origin = self.calculate_policy_adv(
                i, ["tmp", "original", "tmp", "origin"]
            )
            adv_tmp_origin_tmp_tmp = self.calculate_policy_adv(
                i, ["tmp", "original", "tmp", "tmp"]
            )
            adv_tmp_tmp_origin_origin = self.calculate_policy_adv(
                i, ["tmp", "tmp", "original", "origin"]
            )
            adv_tmp_tmp_origin_tmp = self.calculate_policy_adv(
                i, ["tmp", "tmp", "original", "tmp"]
            )
            adv_tmp_tmp_tmp_origin = self.calculate_policy_adv(
                i, ["tmp", "tmp", "tmp", "origin"]
            )
            adv_tmp_tmp_tmp_tmp = self.calculate_policy_adv(
                i, ["tmp", "tmp", "tmp", "tmp"]
            )

            A = np.array(
                [
                    [
                        [
                            [0, adv_origin_origin_origin_tmp],
                            [adv_origin_origin_tmp_origin, adv_origin_origin_tmp_tmp],
                        ],
                        [
                            [adv_origin_tmp_origin_origin, adv_origin_tmp_origin_tmp],
                            [adv_origin_tmp_tmp_origin, adv_origin_tmp_tmp_tmp],
                        ],
                    ],
                    [
                        [
                            [adv_tmp_origin_origin_origin, adv_tmp_origin_origin_tmp],
                            [adv_tmp_origin_tmp_origin, adv_tmp_origin_tmp_tmp],
                        ],
                        [
                            [adv_tmp_tmp_origin_origin, adv_tmp_tmp_origin_tmp],
                            [adv_tmp_tmp_tmp_origin, adv_tmp_tmp_tmp_tmp],
                        ],
                    ],
                ]
            )
            matrix.append(A)
        return matrix

    def compute_payoff_matrix(self):
        if self.num_agent == 2:
            return self.two_player_matrix_constructor()
        if self.num_agent == 3:
            return self.three_player_matrix_constructor()
        if self.num_agent == 4:
            return self.four_player_matrix_constructor()

    def calculate_importance_ratio(self, agent_idx, data, policies):
        importance_ratio = []

        for i in range(self.num_agent):
            if i != agent_idx:
                logp_tmp = data[-1][:, i]
                obs = data[0][:, i]
                act = data[1][:, i]
                logp_ratio = self.agents[i].get_logp_ratio(
                    self.sess, obs, act, logp_tmp, policies[i]
                )
                importance_ratio.append(logp_ratio)
        return np.mean(importance_ratio, axis=1)

    def compute_br_policies(self):
        data_tmp = self.buf.get_tmp()

        for i in range(self.num_agent):
            i_ratio = self.calculate_importance_ratio(
                i, data_tmp, ["tmp" for i in range(self.num_agent)]
            )
            agent_data = [item[:, i] for item in data_tmp]
            agent = self.agents[i]
            p_loss = agent.br_update(self.sess, agent_data, i_ratio)
            self.logger.store(
                {"br_policy_loss": p_loss, "br_importance_rate": i_ratio,}, [i]
            )

    def train(self):

        for epoch in range(self.config["epochs"]):
            # update the tmp policies of agents
            self.rollouts(1, ["original"] * self.num_agent)
            self.update_tmp_policies()
            # use the tmo policies to construct the meta game payoff matrix
            # update the original policies with the nash pair outputed by nash solver
            self.update_policies_with_nash()
            if self.config["br"]:
                self.compute_br_policies()
            # # Log info about epoch
            self.logger.dump(
                self.keys,
                agents=[i for i in range(len(self.agents))],
                step=epoch,
                mean_only=True,
            )
            if epoch % self.config["save_frequency"] == 0:
                print("--- the model has been saved ---")
                self.save(step=epoch)
