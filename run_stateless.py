from matrix_games.stateless_games import plot_dynamics, create_game
from matrix_games.iga_trpo import IGA_TRPO
from matrix_games.iga_pp import IGA_PP
from matrix_games.wolf import WoLF_IGA
from matrix_games.iga import IGA
import nashpy as nash
from matrix_games.util import variety_of_meta_strategy, convergence_rate
import numpy as np
import math


def run(algo, init_policies, payoffs, utilities, config):
    meta_strategy_score = []
    pi_alpha, pi_beta, _, _, meta_strategies, converge_step = IGA_TRPO(
        init_policies[0],
        init_policies[1],
        payoffs[0],
        payoffs[1],
        utilities[0],
        utilities[1],
        config,
    )
    num_strategy, variety_score = variety_of_meta_strategy(meta_strategies)
    if (
        not math.isnan(variety_score)
        and variety_score != float("Inf")
        and variety_score != float("-Inf")
    ):
        meta_strategy_score = [len(num_strategy), variety_score]
    return [pi_alpha[-1], pi_beta[-1]], meta_strategy_score, converge_step


def valid_mean(data):
    sum_value = np.sum(data)
    len_valid = len([i for i in data if i != 0])
    if len_valid == 0:
        return 0
    return sum_value / len_valid


def main():
    games = ["coordination", "anticoordination", "cyclic", "dominance_solvable"]
    iteration = 1000
    lr = 0.02
    kl_coeff = 100

    config = {"iteration": iteration, "lr": lr, "kl_coeff": kl_coeff, "br_lr": 0.03}

    for game_type in games:
        print("------------------{}-----------------".format(game_type))
        trpo_policies = []
        trpo_meta_strategies_scores = []
        trpo_converge_step = []
        adv_policies = []
        adv_meta_strategies_scores = []
        adv_converge_step = []
        diff_policies = []
        diff_meta_strategies_scores = []
        diff_converge_step = []
        wolf_policies = []
        wolf_converge_step = []
        pp_policies = []
        pp_converge_step = []
        iga_policies = []
        iga_converge_step = []
        for i in range(200):
            seed = i
            payoffs, us = create_game(game_type, seed)
            game_name = "asymmetric_{}".format(i)
            init_pi_alpha = 0.9
            init_pi_beta = 0.2
            game = nash.Game(payoffs[0], payoffs[1])
            nash_pairs = []
            for eq in game.support_enumeration():
                nash_pair1 = eq
                nash_pairs.append([nash_pair1[0][0], nash_pair1[1][0]])
            if len(nash_pairs) == 2 or len(nash_pairs) == 4:
                continue

            config.update({"target_nash": nash_pairs})
            policies, meta_strategy_scores, converge_step = run(
                "trpo", [init_pi_alpha, init_pi_beta], payoffs, us, config
            )
            trpo_policies.append(policies)
            trpo_meta_strategies_scores.append(meta_strategy_scores)
            trpo_converge_step.append(converge_step)

            config.update({"lr_max": 0.06, "lr_min": 0.01})
            pi_alpha, pi_beta, _, _, converge_step = WoLF_IGA(
                init_pi_alpha,
                init_pi_beta,
                payoffs[0],
                payoffs[1],
                us[0],
                us[1],
                config,
            )
            wolf_policies.append([pi_alpha[-1], pi_beta[-1]])
            if converge_step > 0:
                wolf_converge_step.append(converge_step)

            config.update({"gamma": 0.01, "single": False})
            pi_alpha, pi_beta, _, _, converge_step = IGA_PP(
                init_pi_alpha,
                init_pi_beta,
                payoffs[0],
                payoffs[1],
                us[0],
                us[1],
                config,
            )
            pp_policies.append([pi_alpha[-1], pi_beta[-1]])
            if converge_step > 0:
                pp_converge_step.append(converge_step)

            config.update({"gamma": 0.01, "single": False})
            pi_alpha, pi_beta, _, _, converge_step = IGA(
                init_pi_alpha,
                init_pi_beta,
                payoffs[0],
                payoffs[1],
                us[0],
                us[1],
                config,
            )
            iga_policies.append([pi_alpha[-1], pi_beta[-1]])
            if converge_step > 0:
                iga_converge_step.append(converge_step)

        print("-------------------convergence rate ---------------------")
        trpo_policies = np.array(trpo_policies)
        wolf_policies = np.array(wolf_policies)
        pp_policies = np.array(pp_policies)

        trpo_conv_rate = len([i for i in trpo_converge_step if i > 0]) / len(
            wolf_converge_step
        )

        adv_conv_rate = len([i for i in adv_converge_step if i > 0]) / len(
            wolf_converge_step
        )
        diff_conv_rate = len([i for i in diff_converge_step if i > 0]) / len(
            wolf_converge_step
        )

        wolf_conv_rate = len([i for i in wolf_converge_step if i > 0]) / len(
            wolf_converge_step
        )

        pp_conv_rate = len([i for i in pp_converge_step if i > 0]) / len(
            wolf_converge_step
        )
        iga_conv_rate = len([i for i in iga_converge_step if i > 0]) / len(
            wolf_converge_step
        )
        print(
            "trpo: ",
            trpo_conv_rate,
            " || average step to converge: ",
            valid_mean(trpo_converge_step),
        )
        print(
            "wolf: ",
            wolf_conv_rate,
            " || average step to converge: ",
            valid_mean(wolf_converge_step),
        )
        print(
            "pp: ",
            pp_conv_rate,
            " ||average step to converge: ",
            valid_mean(pp_converge_step),
        )
        print(
            "iga: ",
            iga_conv_rate,
            " ||average step to converge: ",
            valid_mean(iga_converge_step),
        )

        print("--------------------meta strategy variety------------------")
        trpo_meta_strategies_scores = np.array(trpo_meta_strategies_scores)
        print(
            "trpo average number of strategy: ",
            np.mean(len(trpo_meta_strategies_scores[:][0])),
            "|| score: ",
            np.mean(trpo_meta_strategies_scores),
        )


if __name__ == "__main__":
    main()
