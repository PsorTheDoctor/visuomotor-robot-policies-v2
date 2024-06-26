"""
Behavior Transformer for hybrid observations.
Code based on miniBET implementation:
https://github.com/notmahi/miniBET
"""
import torch
import time

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.models.behavior_transformer import BehaviorTransformer, GPT, GPTConfig
from robomimic.models.resnet import *


@register_algo_factory_func('mini_bet_hybrid')
def algo_config_to_class(algo_config):
    return MiniBETHybrid, {}


class MiniBETHybrid(PolicyAlgo):
    def _create_networks(self):
        self.conditional = False

        img_feature_dim = 1024
        # state_obs_dim: 19 for lift, 23 for can and square, 53 for tool hang
        state_obs_dim = 0
        for k in ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
            state_obs_dim += self.obs_key_shapes[k][0]
        self.obs_dim = img_feature_dim + state_obs_dim
        self.act_dim = 7
        self.goal_dim = self.obs_dim if self.conditional else 0
        self.horizon = 3

        vision_encoder1 = get_resnet('resnet18')
        vision_encoder1 = replace_bn_with_gn(vision_encoder1)
        self.vision_encoder1 = vision_encoder1.to(self.device)

        vision_encoder2 = get_resnet('resnet18')
        vision_encoder2 = replace_bn_with_gn(vision_encoder2)
        self.vision_encoder2 = vision_encoder2.to(self.device)

        self.bet = BehaviorTransformer(
            obs_dim=self.obs_dim, act_dim=self.act_dim, goal_dim=self.goal_dim,
            gpt_model=GPT(GPTConfig(
                block_size=self.horizon, input_dim=self.obs_dim, n_layer=4, n_head=4, n_embd=72
            )),
            n_clusters=24, kmeans_fit_steps=50
        ).to(self.device)

        self.optimizer = self.bet.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-4, betas=[0.9, 0.95]
        )
        # Clear file before write-up
        with open('results/mini_bet_hybrid.csv', 'w+') as f:
            f.write('loss,time\n')

    @staticmethod
    def process_state_obs_dict_for_training(obs_dict):
        obs_list = list()
        for k in ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
            obs_list.append(obs_dict[k])

        obs_tensor = torch.cat(obs_list, dim=2).to(torch.float32)
        return obs_tensor

    @staticmethod
    def process_state_obs_dict_for_evaluation(obs_dict):
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
        obs_tensor = torch.cat(obs_list, dim=1)
        return obs_tensor

    def train_on_batch(self, batch, epoch, validate=False):
        start = time.time()
        info = super(MiniBETHybrid, self).train_on_batch(batch, epoch, validate=validate)

        obs_state_batch = self.process_state_obs_dict_for_training(batch['obs']).to(self.device)
        obs1_batch = batch['obs']['agentview_image'].to(self.device)
        obs2_batch = batch['obs']['robot0_eye_in_hand_image'].to(self.device)
        act_batch = batch['actions'].to(self.device)

        img1_features = self.vision_encoder1(obs1_batch.flatten(end_dim=1))
        img1_features = img1_features.reshape(*obs1_batch.shape[:2], -1)

        img2_features = self.vision_encoder2(obs2_batch.flatten(end_dim=1))
        img2_features = img2_features.reshape(*obs2_batch.shape[:2], -1)
        features = torch.cat([img1_features, img2_features, obs_state_batch], dim=-1)

        _, loss, loss_dict = self.bet(features, None, act_batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        info['Classification loss'] = loss_dict['classification_loss']
        info['Offset loss'] = loss_dict['offset_loss']
        info['Loss'] = loss_dict['total_loss']

        end = time.time()
        with open('results/mini_bet_hybrid.csv', 'a') as f:
            f.write(f"{info['Loss']},{end - start}\n")

        return info

    def get_action(self, obs_dict, goal_dict=None):
        obs1_tensor = obs_dict['agentview_image'].to(self.device)
        obs2_tensor = obs_dict['robot0_eye_in_hand_image'].to(self.device)
        state_tensor = self.process_state_obs_dict_for_evaluation(obs_dict).to(self.device)

        # obs1_tensor = obs1_tensor.reshape(obs1_tensor.shape[1:])
        # obs2_tensor = obs2_tensor.reshape(obs2_tensor.shape[1:])

        with torch.no_grad():
            img1_features = self.vision_encoder1(obs1_tensor)
            img2_features = self.vision_encoder2(obs2_tensor)
            features = torch.cat([img1_features, img2_features, state_tensor], dim=-1)
            features = features.unsqueeze(0)

            act = self.bet(features, None, None)

        return act[0][0].reshape((1, self.act_dim)).to('cpu')

    def log_info(self, info):
        log = PolicyAlgo.log_info(self, info)
        log['Classification loss'] = info['Classification loss']
        log['Offset loss'] = info['Offset loss']
        log['Loss'] = info['Loss']
        return log
