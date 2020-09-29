from agents.pg_general import PGAgent

# from ddpg_meta.ddpg import DDPG
from trainers.meta_trainer import MetaTrainer
from trainers.indi_trainer import IndiTrainer

TRAINERs = {
    "meta": MetaTrainer,
    "indi": IndiTrainer,
}


def default_config():

    return {
        "exp_name": "ppo_indi",
        "trainer": "indi",
        "algo": "ppo",
        "clip_ratio": 0.1,
        "cpu": 1,
        "entropy_coeff": 0.01,
        "env": "PongDuel-v0",
        "epochs": 100,
        "gamma": 0.99,
        "kl_coeff": 0.001,
        "lam": 0.97,
        "local_steps_per_epoch": 2000,
        "max_ep_len": 2000,
        "meta_rollout_epochs": 1,
        "pi_lr": 0.0001,
        "seed": 0,
        "steps_per_epoch": 2000,
        "target_kl": 0.01,
        "train_pi_iters": 10,
        "train_v_iters": 10,
        "model": {"activation": "tanh", "hidden_sizes": [128, 128]},
        "output_dir": "indi_pong_trail1",
        "save_freq": 1,
    }


def update_config(configs):
    default = default_config()
    for k, v in configs.items():
        default[k] = v
    return default
