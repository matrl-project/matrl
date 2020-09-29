from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.logx import EpochLogger
import os
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, config):
        self.log_dir = config["output_dir"]
        logger_kwargs = setup_logger_kwargs(config["exp_name"], config["seed"])
        logger_kwargs["output_dir"] = config["output_dir"]
        self.csv_logger = EpochLogger(**logger_kwargs)
        self.csv_logger.save_config(config)
        self.tf_logger = SummaryWriter(os.path.join(self.log_dir))
        # self.tf_logger.set_as_default()

    def store(self, data_dict, agent):
        for k, v in data_dict.items():
            if len(agent) > 1:
                for i in agent:
                    key = "{}_{}".format(k, i)
                    k_v = {key: v[i]}
                    self.csv_logger.store(**k_v)
            else:
                key = "{}_{}".format(k, agent[0])
                k_v = {key: v}
                self.csv_logger.store(**k_v)

    def dump(self, keys, agents, step, mean_only):
        for k in keys:
            for i in agents:
                key = "{}_{}".format(k, i)
                value = self.csv_logger.epoch_dict[key]

                self.csv_logger.log_tabular(key, average_only=mean_only)

                if mean_only:
                    self.tf_logger.add_scalar(key, np.mean(value), step)
                    # self.tf_logger.add_scalar(scalar)
                else:
                    for p, q in zip(
                        ["min", "mean", "max", "std"],
                        [np.min(value), np.mean(value), np.max(value), np.std(value)],
                    ):
                        # scalar = tf.compat.v1.summary.scalar("{}_{}".format(key, p), q, step)
                        self.tf_logger.add_scalar("{}_{}".format(key, p), q, step)

        self.csv_logger.dump_tabular()
        self.tf_logger.flush()
