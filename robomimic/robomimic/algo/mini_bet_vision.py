"""
Behavior Transformer for hybrid observations.
Code based on miniBET implementation:
https://github.com/notmahi/miniBET
"""
import torch
import time

from diffusers.optimization import get_scheduler

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.models.behavior_transformer import BehaviorTransformer, GPT, GPTConfig
from robomimic.models.resnet import *


@register_algo_factory_func('mini_bet_vision')
def algo_config_to_class(algo_config):
    return MiniBETVision, {}


class MiniBETVision(PolicyAlgo):
    def _create_networks(self):
        self.epochs = 2000
        self.conditional = False
        self.obs_dim = 1024
        self.act_dim = 7
        self.goal_dim = self.obs_dim if self.conditional else 0
        self.horizon = 3  # horizon

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
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=500 * self.epochs  # len(dataloader) * epochs
        )
        # Clear file before write-up
        with open('results/mini_bet_vision.csv', 'w+') as f:
            f.write('loss,time\n')

        self.n_since_last_inference = 0
        self.act_pred = None

    def train_on_batch(self, batch, epoch, validate=False):
        start = time.time()
        info = super(MiniBETVision, self).train_on_batch(batch, epoch, validate=validate)

        obs1_batch = batch['obs']['agentview_image'].to(self.device)
        obs2_batch = batch['obs']['robot0_eye_in_hand_image'].to(self.device)
        act_batch = batch['actions'].to(self.device)

        # obs1_batch = obs1_batch.reshape(obs1_batch.shape[1:])
        # obs2_batch = obs2_batch.reshape(obs2_batch.shape[1:])

        img1_features = self.vision_encoder1(obs1_batch.flatten(end_dim=1))
        img1_features = img1_features.reshape(*obs1_batch.shape[:2], -1)

        img2_features = self.vision_encoder2(obs2_batch.flatten(end_dim=1))
        img2_features = img2_features.reshape(*obs2_batch.shape[:2], -1)
        features = torch.cat([img1_features, img2_features], dim=-1)

        _, loss, loss_dict = self.bet(features, None, act_batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        info['Classification loss'] = loss_dict['classification_loss']
        info['Offset loss'] = loss_dict['offset_loss']
        info['Loss'] = loss_dict['total_loss']

        end = time.time()
        with open('results/mini_bet_vision.csv', 'a') as f:
            f.write(f"{info['Loss']},{end - start}\n")

        return info

    def get_action(self, obs_dict, goal_dict=None):
        if True:  # self.n_since_last_inference % self.horizon == 0:
            # Pretend batch size to be 1
            obs1_tensor = obs_dict['agentview_image'].to(self.device)
            obs2_tensor = obs_dict['robot0_eye_in_hand_image'].to(self.device)

            # obs1_tensor = obs1_tensor.reshape(obs1_tensor.shape[1:])
            # obs2_tensor = obs2_tensor.reshape(obs2_tensor.shape[1:])

            with torch.no_grad():
                img1_features = self.vision_encoder1(obs1_tensor)
                img2_features = self.vision_encoder2(obs2_tensor)
                features = torch.cat([img1_features, img2_features], dim=-1)
                features = features.unsqueeze(0)

                act, _, _ = self.bet(features, None, None)

            # self.act_pred = act.detach().to('cpu').numpy()
            # self.n_since_last_inference = (self.n_since_last_inference + 1) % self.horizon

        return act[0][0].reshape((1, self.act_dim)).to('cpu')  # self.act_pred[:, self.n_since_last_inference]

    def log_info(self, info):
        log = PolicyAlgo.log_info(self, info)
        log['Classification loss'] = info['Classification loss']
        log['Offset loss'] = info['Offset loss']
        log['Loss'] = info['Loss']
        return log
