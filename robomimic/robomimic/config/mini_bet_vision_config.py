from robomimic.config.base_config import BaseConfig


class MiniBETVisionConfig(BaseConfig):
    ALGO_NAME = 'mini_bet_vision'

    def algo_config(self):
        self.algo.optim_params.policy.learning_rate.initial = 1e-5
        self.algo.optim_params.policy.decay_factor = 2e-4
