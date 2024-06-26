from robomimic.config.base_config import BaseConfig


class DPVisionConfig(BaseConfig):
    ALGO_NAME = 'dp_vision'

    def algo_config(self):
        # self.algo.optim_params.policy.optimizer_type = 'adam'  # 'adamw'
        self.algo.optim_params.policy.learning_rate.initial = 1e-4
        self.algo.optim_params.policy.decay_factor = 1e-6

    # def train_config(self):
    #     self.train.num_epochs = 100
