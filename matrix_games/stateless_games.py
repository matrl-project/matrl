import numpy as np
import matplotlib

# import matplotlib.pyplot as plt
from copy import deepcopy

FONTSIZE = 12


def plot_dynamics(
    history_pi_0,
    history_pi_1,
    pi_alpha_gradient_history,
    pi_beta_gradient_history,
    nash_points,
    title="",
):
    """this function plots the policy changing curves with x and y axis in range [0, 1]
    
    Arguments:
        history_pi_0 {[]float} -- list of policy values of player 1
        history_pi_1 {[]float} --list of policy values of player 2
        pi_alpha_gradient_history {[]float} -- list of gradient values of player 1
        pi_beta_gradient_history {[]float} --  list of gradient values of player 2
    
    Keyword Arguments:
        title {str} -- the title for the draw (default: {''})
    """
    cmap = plt.get_cmap("viridis")
    colors = range(len(history_pi_1))
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    scatter = ax.scatter(history_pi_0, history_pi_1, c=colors, s=1)
    for pair in nash_points:
        ax.scatter(pair[0], pair[1], c="r", s=20.0, marker="*")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Iterations", rotation=270, fontsize=FONTSIZE)

    skip = slice(0, len(history_pi_0), 50)
    ax.quiver(
        history_pi_0[skip],
        history_pi_1[skip],
        pi_alpha_gradient_history[skip],
        pi_beta_gradient_history[skip],
        units="xy",
        scale=10.0,
        zorder=3,
        color="blue",
        width=0.007,
        headwidth=3.0,
        headlength=4.0,
    )

    ax.set_ylabel("Policy of Player 2", fontsize=FONTSIZE)
    ax.set_xlabel("Policy of Player 1", fontsize=FONTSIZE)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 1.0)
    ax.set_title(title, fontsize=FONTSIZE + 8)
    plt.tight_layout()
    plt.savefig("{}.png".format(title), dpi=fig.dpi)
    plt.show()


GAMES = {
    # nash 0.5，0.5
    "rotation": [np.array([[0, 3], [1, 2]]), np.array([[3, 2], [0, 1]])],
    #  nash 1，1
    "prisoner_dilema": [np.array([[3, 1], [4, 2]]), np.array([[3, 4], [1, 2]])],
    # nash 0.5, 0,.5
    "matching_pennies": [np.array([[1, -1], [-1, 1]]), np.array([[-1, 1], [1, -1]])],
    "stag_hunt": [np.array([[4, 1], [3, 2]]), np.array([[4, 3], [1, 2]])],
    "chicken": [np.array([[3, 2], [4, 1]]), np.array([[3, 4], [2, 1]])],
    "deadlock": [np.array([[1, 2], [3, 4]]), np.array([[1, 3], [2, 4]])],
    "harmony": [np.array([[4, 3], [2, 1]]), np.array([[4, 2], [3, 1]])],
    "coordination_1": [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])],
    "coordination_2": [np.array([[4, 2], [1, 3]]), np.array([[3, 2], [1, 4]])],
}


def symmetric_random(num_players, strategies):
    payoff0 = np.random.random_sample((num_players,) + tuple([strategies,]))
    payoff1 = np.array([[payoff0[0][0], payoff0[1][0]], [payoff0[0][1], payoff0[1][1]]])
    return payoff0, payoff1


def asymmetric_random(num_players, strategies):
    return (
        np.random.random_sample((num_players,) + tuple([strategies,])),
        np.random.random_sample((num_players,) + tuple([strategies,])),
    )


def coordination_game():
    b = np.random.random_sample()
    d = b + np.random.random_sample()
    c = d + np.random.random_sample()
    a = c + np.random.random_sample()
    g = np.random.random_sample()
    e = g + np.random.random_sample()
    f = np.random.random_sample()
    h = f + np.random.random_sample()
    assert a > c and b < d and e > g and f < h and b < a and d < c and a > 0 and c > 0
    return np.array([[a, b], [c, d]]), np.array([[e, g], [f, h]])


def anticoordination_game():
    a = -np.random.random_sample() - 1
    c = a + np.random.random_sample()
    if np.random.random_sample() < 0.5:
        b = abs(a) - np.random.random_sample() * 0.1
        if abs(c) - b < 0:
            d = abs(c) - np.random.random_sample() * 0.1
        else:
            d = b - np.random.random_sample() * 0.1
    else:
        d = c + np.random.random_sample() * 0.1
        b = d + np.random.random_sample() * 0.1

    e = np.random.random_sample()
    g = e + np.random.random_sample()
    h = np.random.random_sample()
    f = h + np.random.random_sample()
    assert (
        a < c
        and b > d
        and e < g
        and f > h
        and abs(b) < abs(a)
        and abs(d) < abs(c)
        and a < 0
        and c < 0
    ), "{}".format([a, b, c, d, e, f, g, h])
    return np.array([[a, b], [c, d]]), np.array([[e, g], [f, h]])


def cyclic_game():
    if np.random.random_sample() < 0.5:
        e = np.random.random_sample()
        g = e + np.random.random_sample()
        h = np.random.random_sample()
        f = h + np.random.random_sample()
        a = 0.5 + np.random.random_sample()
        c = -a + np.random.random_sample() * 0.1
        d = abs(c) - np.random.random_sample() * 0.1
        b = d - np.random.random_sample() * 0.1
        assert (
            a > c
            and e < g
            and b < d
            and f > h
            and abs(b) < abs(a)
            and abs(d) < abs(c)
            and a * c < 0
        ), "{}".format([a, b, c, d, e, f, g, h])
    else:
        g = np.random.random_sample()
        e = g + np.random.random_sample()
        f = np.random.random_sample()
        h = f + np.random.random_sample()
        c = np.random.random_sample() + 0.5
        a = -c + np.random.random_sample() * 0.1
        b = abs(a) - np.random.random_sample() * 0.1
        d = b - np.random.random_sample() * 0.1
        assert (
            a < c
            and e > g
            and b > d
            and f < h
            and abs(b) < abs(a)
            and abs(d) < abs(c)
            and a * c < 0
        ), "{}".format([a, b, c, d, e, f, g, h])
    return np.array([[a, b], [c, d]]), np.array([[e, g], [f, h]])


def dominance_solvable():
    e = np.random.random_sample()
    g = np.random.random_sample()
    f = np.random.random_sample()
    h = np.random.random_sample()
    a = np.random.random_sample()
    c = np.random.random_sample()
    if np.random.random_sample() < 0.5:
        b = a + np.random.random_sample()
        d = np.random.random_sample()
    else:
        d = c + np.random.random_sample()
        b = np.random.random_sample()
    assert b > a or d > c
    return np.array([[a, b], [c, d]]), np.array([[e, g], [f, h]])


def create_game(name, seed):
    """this function create the payoff matrix of the given game
    
    Arguments:
        name {string} -- the name of the game
    
    Returns:
        [][]float -- the payoff matrix for player1 and player2
        []float -- the utility values of player1 and player2
    """
    np.random.seed(seed)
    if name == "symmetric_random":
        payoff_0, payoff_1 = symmetric_random(2, 2)
    elif name == "asymmetric_random":
        payoff_0, payoff_1 = asymmetric_random(2, 2)
    elif name == "coordination":
        payoff_0, payoff_1 = coordination_game()
    elif name == "anticoordination":
        payoff_0, payoff_1 = anticoordination_game()
    elif name == "cyclic":
        payoff_0, payoff_1 = cyclic_game()
    elif name == "dominance_solvable":
        payoff_0, payoff_1 = dominance_solvable()
    else:
        payoff_0, payoff_1 = GAMES[name][0], GAMES[name][1]
    u_alpha = payoff_0[(0, 0)] - payoff_0[(0, 1)] - payoff_0[(1, 0)] + payoff_0[(1, 1)]
    u_beta = payoff_1[(0, 0)] - payoff_1[(0, 1)] - payoff_1[(1, 0)] + payoff_1[(1, 1)]
    return [payoff_0, payoff_1], [u_alpha, u_beta]
