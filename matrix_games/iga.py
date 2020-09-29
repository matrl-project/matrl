from matrix_games.util import V, A, A_TR, convergence_check
from matrix_games.iga_trpo import kl


def IGA(pi_alpha, pi_beta, payoff_0, payoff_1, u_alpha, u_beta, config):
    pi_alpha_history = [pi_alpha]
    pi_beta_history = [pi_beta]
    pi_alpha_gradient_history = [0.0]
    pi_beta_gradient_history = [0.0]
    converge_step = 0
    for i in range(config["iteration"]):
        pi_alpha_gradient = pi_beta * u_alpha + payoff_0[(0, 1)] - payoff_0[(1, 1)]
        pi_beta_gradient = pi_alpha * u_beta + payoff_1[(1, 0)] - payoff_1[(1, 1)]
        pi_alpha_next = pi_alpha + config["lr"] * pi_alpha_gradient
        pi_beta_next = pi_beta + config["lr"] * pi_beta_gradient
        pi_alpha = max(0.0, min(1.0, pi_alpha_next))
        pi_beta = max(0.0, min(1.0, pi_beta_next))
        ########### END TODO ###############################
        pi_alpha_gradient_history.append(pi_alpha_gradient)
        pi_beta_gradient_history.append(pi_beta_gradient)
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
        converge_step,
    )
