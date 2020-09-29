from matrix_games.util import V, A, A_TR, convergence_check


def WoLF_IGA(pi_alpha, pi_beta, payoff_0, payoff_1, u_alpha, u_beta, config):

    pi_alpha_history = [pi_alpha]
    pi_beta_history = [pi_beta]
    pi_alpha_gradient_history = [0.0]
    pi_beta_gradient_history = [0.0]
    converge_step = 0
    for i in range(config["iteration"]):
        lr_alpha = config["lr_max"]
        lr_beta = config["lr_max"]

        if V(pi_alpha, pi_beta, payoff_0) > V(
            config["target_nash"][0][0], pi_beta, payoff_0
        ):
            lr_alpha = config["lr_min"]
        if V(pi_alpha, pi_beta, payoff_1) > V(
            pi_alpha, config["target_nash"][0][1], payoff_1
        ):
            lr_beta = config["lr_min"]
        pi_alpha_gradient = pi_beta * u_alpha + payoff_0[(0, 1)] - payoff_0[(1, 1)]
        pi_beta_gradient = pi_alpha * u_beta + payoff_1[(1, 0)] - payoff_1[(1, 1)]
        pi_alpha_next = pi_alpha + lr_alpha * pi_alpha_gradient
        pi_beta_next = pi_beta + lr_beta * pi_beta_gradient
        pi_alpha = max(0.0, min(1.0, pi_alpha_next))
        pi_beta = max(0.0, min(1.0, pi_beta_next))
        pi_alpha_gradient_history.append(pi_alpha_gradient)
        pi_beta_gradient_history.append(pi_beta_gradient)
        pi_alpha_history.append(pi_alpha)
        pi_beta_history.append(pi_beta)
        if converge_step == 0 and convergence_check(
            pi_alpha, pi_beta, config["target_nash"]
        ):
            converge_step = i
    return (
        pi_alpha_history,
        pi_beta_history,
        pi_alpha_gradient_history,
        pi_beta_gradient_history,
        converge_step,
    )


def WoLF_IGA3(pi_alpha, pi_beta, payoff_0, payoff_1, u_alpha, u_beta, config):

    pi_alpha_history = [pi_alpha]
    pi_beta_history = [pi_beta]
    pi_alpha_gradient_history = [0.0]
    pi_beta_gradient_history = [0.0]
    converge_step = 0
    for i in range(config["iteration"]):
        lr_alpha = config["lr_max"]
        lr_beta = config["lr_max"]

        if V(pi_alpha, pi_beta, payoff_0) > V(
            config["target_nash"][0][0], pi_beta, payoff_0
        ):
            lr_alpha = config["lr_min"]
        if V(pi_alpha, pi_beta, payoff_1) > V(
            pi_alpha, config["target_nash"][0][1], payoff_0
        ):
            lr_beta = config["lr_min"]
        pi_alpha_gradient = pi_beta * u_alpha + payoff_0[(0, 1)] - payoff_0[(1, 1)]
        pi_beta_gradient = pi_alpha * u_beta + payoff_1[(1, 0)] - payoff_1[(1, 1)]
        pi_alpha_next = pi_alpha + lr_alpha * pi_alpha_gradient
        pi_beta_next = pi_beta + lr_beta * pi_beta_gradient
        pi_alpha = max(0.0, min(1.0, pi_alpha_next))
        pi_beta = max(0.0, min(1.0, pi_beta_next))
        pi_alpha_gradient_history.append(pi_alpha_gradient)
        pi_beta_gradient_history.append(pi_beta_gradient)
        pi_alpha_history.append(pi_alpha)
        pi_beta_history.append(pi_beta)
        if converge_step == 0 and convergence_check(
            pi_alpha, pi_beta, config["target_nash"]
        ):
            converge_step = i
    return (
        pi_alpha_history,
        pi_beta_history,
        pi_alpha_gradient_history,
        pi_beta_gradient_history,
        converge_step,
    )
