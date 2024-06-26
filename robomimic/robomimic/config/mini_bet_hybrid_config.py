from robomimic.config.base_config import BaseConfig


class MiniBETHybridConfig(BaseConfig):
    ALGO_NAME = 'mini_bet_hybrid'

    def algo_config(self):
        self.algo.optim_params.policy.learning_rate.initial = 1e-4
        self.algo.optim_params.policy.decay_factor = 0.1
