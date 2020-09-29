import numpy as np
import nashpy as nash
from matrix_games.util import V, A, A_TR, convergence_check


def kl(p, q):
    """calculate the 
     divergence of policy p and q 
    in our case we have discrete policy over two discrete states 
    
    Arguments:
        p {[]float} -- policy p 
        q {[]float} -- policy q 

    Returns:
        float -- the kl divergence value
    """
    if p == 0 or q == 0 or p == 1 or q == 1:
        return 0
    kl_r = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return kl_r


def get_meta_game_with_trpo(a, b, a_next, b_next, payoff_0, payoff_1):
    # pi(a |s) A(a,s)
    A = [
        [0, A_TR(a, b_next, a, b, payoff_0)],
        [A_TR(a_next, b, a, b, payoff_0), A_TR(a_next, b_next, a, b, payoff_0)],
    ]
    B = [
        [0, A_TR(a, b_next, a, b, payoff_1)],
        [A_TR(a_next, b, a, b, payoff_1), A_TR(a_next, b_next, a, b, payoff_1)],
    ]
    return A, B


def get_next_policy_trpo(
    pi_alpha, pi_beta, u_alpha, u_beta, payoff_0, payoff_1, lr, kl_coeff
):
    policies_wo_kl = []
    pi_alpha_gradient = pi_beta * u_alpha + payoff_0[(0, 1)] - payoff_0[(1, 1)]
    pi_beta_gradient = pi_alpha * u_beta + payoff_1[(1, 0)] - payoff_1[(1, 1)]
    pi_alpha_next = pi_alpha + lr * pi_alpha_gradient
    pi_beta_next = pi_beta + lr * pi_beta_gradient
    policies_wo_kl = [pi_alpha_next, pi_beta_next]
    kl_a = kl(pi_alpha, pi_alpha_next)
    kl_b = kl(pi_beta, pi_beta_next)

    pi_alpha_next = pi_alpha + lr * (pi_alpha_gradient + kl_coeff * kl_a)
    pi_beta_next = pi_beta + lr * (pi_beta_gradient + kl_coeff * kl_b)
    return (
        pi_alpha_next,
        pi_beta_next,
        pi_alpha_gradient,
        pi_beta_gradient,
        policies_wo_kl,
    )


def get_nash_next(a0, b0, a1, b1, payoff_0, payoff_1, kl_coeff):
    A, B = get_meta_game_with_trpo(a0, b0, a1, b1, payoff_0, payoff_1)
    #     meta_game = nash.Game(A, B) # play the game
    game1 = nash.Game(A, B)

    # find the NE for the game with the updated pay off matrix
    for eq in game1.support_enumeration():
        nash_pair1 = eq

    # break
    a = nash_pair1[0][0] * a0 + nash_pair1[0][1] * a1 + kl_coeff * kl(a0, a1)
    b = nash_pair1[1][0] * b0 + nash_pair1[1][1] * b1 + kl_coeff * kl(b0, b1)

    return a, b, [nash_pair1[0][0], nash_pair1[1][0]]


def IGA_TRPO(pi_alpha, pi_beta, payoff_0, payoff_1, u_alpha, u_beta, config):
    pi_alpha_history = []
    pi_beta_history = []
    pi_alpha_gradient_history = []
    pi_beta_gradient_history = []
    meta_strategies = []
    converge_step = 0
    pi_alpha_raw = []
    pi_beta_raw = []
    pi_alpha_no_meta = []
    pi_beta_no_meta = []
    for i in range(config["iteration"]):
        # calculate the gradient
        pi_alpha_1, pi_beta_1, grad_alpha, grad_beta, pi_wo_kl = get_next_policy_trpo(
            pi_alpha,
            pi_beta,
            u_alpha,
            u_beta,
            payoff_0,
            payoff_1,
            config["lr"],
            config["kl_coeff"],
        )
        pi_alpha_raw.append(pi_wo_kl[0])
        pi_beta_raw.append(pi_wo_kl[1])
        # clip the probability, this is the range of  probability,
        pi_alpha_1 = max(0.0, min(1.0, pi_alpha_1))
        pi_beta_1 = max(0.0, min(1.0, pi_beta_1))
        pi_alpha_no_meta.append(pi_alpha_1)
        pi_beta_no_meta.append(pi_beta_1)
        pi_alpha_next, pi_beta_next, meta_strategy = get_nash_next(
            pi_alpha,
            pi_beta,
            pi_alpha_1,
            pi_beta_1,
            payoff_0,
            payoff_1,
            config["kl_coeff"],
        )
        meta_strategies.append(meta_strategy)
        BR_pi_alpha_gradient = (
            pi_beta_next * u_alpha + payoff_0[(0, 1)] - payoff_0[(1, 1)]
        )
        BR_pi_beta_gradient = (
            pi_alpha_next * u_beta + payoff_1[(1, 0)] - payoff_1[(1, 1)]
        )
        BR_pi_alpha_next = pi_alpha + config["br_lr"] * BR_pi_alpha_gradient
        BR_pi_beta_next = pi_beta + config["br_lr"] * BR_pi_beta_gradient
        pi_alpha = max(0.0, min(1.0, BR_pi_alpha_next))
        pi_beta = max(0.0, min(1.0, BR_pi_beta_next))
        pi_alpha_gradient_history.append(grad_alpha)
        pi_beta_gradient_history.append(grad_beta)
        pi_alpha_history.append(pi_alpha)
        pi_beta_history.append(pi_beta)
        if converge_step == 0 and convergence_check(
            pi_alpha, pi_beta, config["target_nash"]
        ):
            converge_step = i
            break

    return (
        pi_alpha_history,
        pi_beta_history,
        pi_alpha_gradient_history,
        pi_beta_gradient_history,
        meta_strategies,
        converge_step,
        # pi_alpha_raw,
        # pi_beta_raw,
        # pi_alpha_no_meta,
        # pi_beta_no_meta,
    )
