import numpy as np


def V(alpha, beta, payoff):
    """The function calculates the  expected payoff of given strategies alpha and beta 
    
    Arguments:
        alpha {float} -- player 1 's policy
        beta {float} -- player 2 's policy
        payoff {[][]float]} -- the payoff function of the game
    
    Returns:
        float -- the value of the state 
    """
    u = payoff[(0, 0)] - payoff[(0, 1)] - payoff[(1, 0)] + payoff[(1, 1)]
    v = (
        alpha * beta * u
        + alpha * (payoff[(0, 1)] - payoff[(1, 1)])
        + beta * (payoff[(1, 0)] - payoff[(1, 1)])
        + payoff[(1, 1)]
    )
    return v


def A(a1, a2, alpha, beta, payoff):
    v = V(alpha, beta, payoff)
    adv = payoff[(a1, a2)] - v
    return adv


def A_TR(alpha_p, beta_p, alpha, beta, payoff):
    a00 = (alpha_p * beta_p - alpha * beta) * A(0, 0, alpha, beta, payoff)
    a01 = (alpha_p * (1 - beta_p) - alpha * (1 - beta_p)) * A(0, 1, alpha, beta, payoff)
    a10 = ((1 - alpha_p) * beta - (1 - alpha) * beta) * A(1, 0, alpha, beta, payoff)
    a11 = ((1 - alpha_p) * (1 - beta_p) - (1 - alpha) * (1 - beta)) * A(
        1, 1, alpha, beta, payoff
    )
    a_tr = a00 + a01 + a10 + a11
    return a_tr


def is_close(a, b):
    if (a - b) ** 2 < 2.5 * 1e-3:
        return True
    return False


def convergence_check(alpha, beta, nash_pairs):
    """this function checks if the agent is converged to nash
    
    Arguments:
        alpha {float} -- the policy value for action 1 for agent 1
        beta {[type]} -- the policy value for action 2 for agent 2
        nash_pairs {[][]float} -- the nash pairs for the game, 
                the value represent the action 1 for each player
    
    Returns:
        bool -- is the agent converge to nash or not 
    """
    for pair in nash_pairs:
        if is_close(alpha, pair[0]) and is_close(beta, pair[1]):
            return True
    return False


def convergence_rate(alpha_list, beta_list, nash_pair_list):
    """ computes the percentage of games converged to nash
    
    Arguments:
        alpha_list {[]float} -- the list of policies of agent 1
        beta_list {[]float} -- the list of policies of agent 2
        nash_pair_list {[][]float} -- the list of nash equilibrium pairs 
    
    Returns:
        float -- the percentage of converged game
    """
    assert len(alpha_list) == len(beta_list)
    converged_num = 0
    for i in range(len(alpha_list)):
        if convergence_check(alpha_list[i], beta_list[i], nash_pair_list):
            converged_num += 1
    return converged_num / len(alpha_list)


def variety_of_meta_strategy(meta_game_nash_pairs):
    """ calculates the variance of meta game strategies 
    
    Arguments:
        meta_game_nash_pairs {[][]} -- the list of meta game strategies
    
    Returns:
        [float] -- the variance 
                    0 if meta strategies are all same 
                    > 1 strategies
    """

    strategies = {}
    strategies[str(meta_game_nash_pairs[0])] = 0
    for pair in meta_game_nash_pairs[1:]:
        if str(pair) in strategies.keys():
            strategies[str(pair)] += 1
        else:
            strategies[str(pair)] = 0

    strategies_percent = [
        value / len(meta_game_nash_pairs) for value in strategies.values()
    ]
    factor = np.std(strategies_percent)
    if len(strategies_percent) == 1:
        factor = 1
    return strategies_percent, np.log(np.exp(len(strategies_percent)) * factor)


def check_all_positive(data):
    for i in range(len(data)):
        if any(i < 0 for i in data[i]):
            return False
    return True
