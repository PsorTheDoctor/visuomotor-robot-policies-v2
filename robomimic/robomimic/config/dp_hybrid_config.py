from robomimic.config.base_config import BaseConfig


class DPHybridConfig(BaseConfig):
    ALGO_NAME = 'dp_hybrid'

    def algo_config(self):
        self.algo.optim_params.policy.learning_rate.initial = 1e-4
        self.algo.optim_params.policy.decay_factor = 1e-6
