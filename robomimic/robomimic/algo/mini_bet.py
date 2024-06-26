"""
Behavior Transformer for low dim observations.
Code based on miniBET implementation:
https://github.com/notmahi/miniBET
"""
import torch
import time

from diffusers.optimization import get_scheduler

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.models.behavior_transformer import BehaviorTransformer, GPT, GPTConfig


@register_algo_factory_func('mini_bet')
def algo_config_to_class(algo_config):
    return MiniBET, {}


class MiniBET(PolicyAlgo):
    def _create_networks(self):
        self.conditional = False
        # obs_dim: 19 for lift, 23 for can and square, 53 for tool hang
        self.obs_dim = 0
        for k in list(self.obs_key_shapes.keys()):
            self.obs_dim += self.obs_key_shapes[k][0]
        self.act_dim = 7
        self.goal_dim = self.obs_dim if self.conditional else 0
        self.horizon = 3  # horizon
        self.epochs = 2000

        self.bet = BehaviorTransformer(
            obs_dim=self.obs_dim, act_dim=self.act_dim, goal_dim=self.goal_dim,
            gpt_model=GPT(GPTConfig(
                block_size=self.horizon, input_dim=self.obs_dim, n_layer=4, n_head=4, n_embd=72
            )),
            n_clusters=24,
            kmeans_fit_steps=50
        ).to(self.device)

        self.optimizer = self.bet.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-4, betas=[0.9, 0.95]
        )
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=500 * self.epochs  # len(dataloader) * epochs
        )
        # Clear file before write-up
        with open('results/mini_bet_state.csv', 'w+') as f:
            f.write('loss,time\n')

    @staticmethod
    def process_obs_dict_for_training(obs_dict):
        obs_list = list()
        for k in list(obs_dict.keys()):
            obs_list.append(obs_dict[k])

        obs_tensor = torch.cat(obs_list, dim=2).to(torch.float32)
        return obs_tensor

    # def process_goal_obs_dict_for_training(self, obs_dict):
    #     obs_list = list()
    #     for k in list(obs_dict.keys()):
    #         obs_list.append(obs_dict[k])
    #
    #     obs_tensor = torch.cat(obs_list, dim=1).to(torch.float32)
    #     # (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
    #     obs_tensor = obs_tensor.reshape((obs_tensor.shape[0], 1, obs_tensor.shape[1]))
    #     return obs_tensor

    @staticmethod
    def process_obs_dict_for_evaluation(obs_dict):
        obs_list = list()
        # It looks like the commented velocity values are available
        # only if seq_length in config file is set to 1.
        obs_keys = [
            'object', 'robot0_eef_pos', 'robot0_eef_quat',  # 'robot0_eef_vel_lin', 'robot0_eef_vel_ang',
            'robot0_gripper_qpos',  # 'robot0_gripper_qvel'
        ]
        for k in obs_keys:
            obs_list.append(obs_dict[k])

        # obs_tensor = torch.zeros((1, 1, self.obs_dim), dtype=torch.float32)
        # # The last dim of obs_list is 3 + 4 + 2 = 9
        # obs_tensor[:, :, :9] = torch.cat(obs_list, dim=2)
        obs_tensor = torch.cat(obs_list, dim=2)
        return obs_tensor

    def train_on_batch(self, batch, epoch, validate=False):
        start = time.time()
        info = super(MiniBET, self).train_on_batch(batch, epoch, validate=validate)

        # for k in list(batch['goal_obs'].keys()):
        #     batch['goal_obs'][k] = batch['goal_obs'][k].reshape((batch['goal_obs'][k].shape[0], 1, batch['goal_obs'][k].shape[1]))

        obs_batch = self.process_obs_dict_for_training(batch['obs']).to(self.device)  # [:, :self.horizon]
        # goal_batch = self.process_obs_dict_for_training(batch['goal_obs']).to(self.device)
        act_batch = batch['actions'].to(self.device)  # [:, :self.horizon]

        _, loss, loss_dict = self.bet(obs_batch, None, act_batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        info['Classification loss'] = loss_dict['classification_loss']
        info['Offset loss'] = loss_dict['offset_loss']
        info['Loss'] = loss_dict['total_loss']

        end = time.time()
        with open('results/mini_bet_state.csv', 'a') as f:
            f.write(f"{info['Loss']},{end - start}\n")

        return info

    def get_action(self, obs_dict, goal_dict=None):
        # Pretend batch size to be 1
        for k in list(obs_dict.keys()):
            obs_dict[k] = obs_dict[k].reshape((1, obs_dict[k].shape[0], obs_dict[k].shape[1]))

        obs_tensor = self.process_obs_dict_for_evaluation(obs_dict)
        goal_tensor = None

        with torch.no_grad():
            act = self.bet(obs_tensor, goal_tensor, None)

        return act[0].reshape((1, self.act_dim)).to('cpu')

    def log_info(self, info):
        log = PolicyAlgo.log_info(self, info)
        log['Classification loss'] = info['Classification loss']
        log['Offset loss'] = info['Offset loss']
        log['Loss'] = info['Loss']
        return log
