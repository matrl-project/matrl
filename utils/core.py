import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None, dim))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None, 1))
    raise NotImplementedError


def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=activation)


def rnn(
    x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None,
):

    x = tf.unstack(x, 5, 1)
    lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(hidden_sizes[0])

    outputs, _ = tf.compat.v1.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    x = tf.layers.dense(outputs[-1], units=hidden_sizes[-1], activation=activation)
    return x


def cnn(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):

    # x shape is (None, width, height, channel)
    x = tf.keras.layers.Conv2D(8, 3, (4, 4), "VALID")(x)
    x = tf.keras.layers.Conv2D(4, 3, (2, 2), "VALID")(x)
    x = tf.keras.layers.Conv2D(3, 3, (1, 1), "VALID")(x)
    x = tf.compat.v1.layers.flatten(x)
    # reshape to policy shape
    x = tf.keras.layers.Dense(hidden_sizes[0], activation=activation)(x)
    x = tf.keras.layers.Dense(hidden_sizes[-1])(x)
    return x


def get_vars(scope=""):
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope=""):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (
        ((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi)
    )
    return tf.reduce_sum(pre_sum, axis=1)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""


def mlp_categorical_policy(
    network, x, a, hidden_sizes, activation, output_activation, action_space
):
    a = tf.squeeze(a, axis=1)
    act_dim = action_space.n
    logits = network(x, list(hidden_sizes) + [act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi


def mlp_gaussian_policy(
    network, x, a, hidden_sizes, activation, output_activation, action_space
):
    act_dim = a.shape.as_list()[-1]
    mu = network(x, list(hidden_sizes) + [act_dim], activation, output_activation)
    log_std = tf.get_variable(
        name="log_std", initializer=-0.5 * np.ones(act_dim, dtype=np.float32)
    )
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


"""
Actor-Critics
"""


def mlp_actor_critic(network, x, a, config, name):

    # default policy builder depends on action space
    config["policy"] = None
    if config["policy"] is None and isinstance(config["action_space"], Box):
        policy = mlp_gaussian_policy
    elif config["policy"] is None and isinstance(config["action_space"], Discrete):
        policy = mlp_categorical_policy
    activation = config["model"]["activation"]
    if activation == "tanh":
        activation_fn = tf.nn.tanh
    if activation == "relu":
        activation_fn = tf.nn.relu
    if activation == "leaky_relu":
        activation_fn = tf.nn.leaky_relu
    output_activation = tf.nn.tanh
    with tf.variable_scope("pi_{}".format(name)):
        pi, logp, logp_pi = policy(
            network,
            x,
            a,
            config["model"]["hidden_sizes"],
            activation_fn,
            output_activation,
            config["action_space"],
        )

    with tf.variable_scope("v_{}".format(name)):
        v = tf.squeeze(
            network(
                x,
                list(config["model"]["hidden_sizes"]) + [1],
                activation_fn,
                config["rnn_length"],
            ),
            axis=1,
        )
    return pi, logp, logp_pi, v
