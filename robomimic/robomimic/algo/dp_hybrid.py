"""
Diffusion Policy for hybrid observations.
Code based on the official implementation:
https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing
"""
import torch
import time

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.models.resnet import *
from robomimic.models.unet import ConditionalUnet1D


@register_algo_factory_func('dp_hybrid')
def algo_config_to_class(algo_config):
    return DPHybrid, {}


class DPHybrid(PolicyAlgo):
    def _create_networks(self):
        self.batch_size = 64
        self.epochs = 2000
        self.diffusion_iters = 100

        self.pred_horizon = 16
        self.obs_horizon = 2
        self.act_horizon = 8

        img_feature_dim = 1024
        # state_obs_dim: 19 for lift, 23 for can and square, 53 for tool hang
        state_obs_dim = 0
        for k in ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
            state_obs_dim += self.obs_key_shapes[k][0]
        self.obs_dim = img_feature_dim + state_obs_dim
        self.act_dim = 7

        vision_encoder1 = get_resnet('resnet18')
        vision_encoder1 = replace_bn_with_gn(vision_encoder1)
        vision_encoder1 = vision_encoder1.to(self.device)

        vision_encoder2 = get_resnet('resnet18')
        vision_encoder2 = replace_bn_with_gn(vision_encoder2)
        vision_encoder2 = vision_encoder2.to(self.device)

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=self.obs_dim * self.obs_horizon
        ).to(self.device)

        self.nets = nn.ModuleDict({
            'vision_encoder1': vision_encoder1,
            'vision_encoder2': vision_encoder2,
            'noise_pred_net': self.noise_pred_net
        }).to(self.device)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.diffusion_iters,
            # beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            # beta_start=0.0001,
            clip_sample=True,
            prediction_type='epsilon',
            # variance_type='fixed_small'
        )
        self.ema = EMAModel(
            parameters=self.nets.parameters(),
            # inv_gamma=1.0,
            # max_value=0.9999,
            # min_value=0.0,
            power=0.75,
            # update_after_step=0
        )
        self.optimizer = torch.optim.AdamW(
            params=self.nets.parameters(),
            # betas=(0.95, 0.999),
            # eps=1e-8,
            lr=1e-4,
            weight_decay=1e-6
        )
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=500 * self.epochs  # len(dataloader) * epochs
        )
        # Clear file before write-up
        with open('results/dp_hybrid.csv', 'w+') as f:
            f.write('loss,time\n')

        self.n_since_last_inference = 0
        self.action_pred = None

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

        # obs_tensor = torch.zeros((1, self.obs_horizon, self.obs_dim), dtype=torch.float32)
        # # The last dim of obs_list is 3 + 4 + 2 = 9
        # obs_tensor[:, :, :9] = torch.cat(obs_list, dim=2)
        obs_tensor = torch.cat(obs_list, dim=2)
        obs_tensor = obs_tensor.reshape(obs_tensor.shape[1], obs_tensor.shape[2])
        return obs_tensor

    def train_on_batch(self, batch, epoch, validate=False):
        start = time.time()
        info = super(DPHybrid, self).train_on_batch(batch, epoch, validate=validate)

        obs1_batch = batch['obs']['agentview_image'][:, :self.obs_horizon].to(self.device)
        obs2_batch = batch['obs']['robot0_eye_in_hand_image'][:, :self.obs_horizon].to(self.device)
        obs_state_batch = self.process_state_obs_dict_for_training(batch['obs'])[:, :self.obs_horizon].to(self.device)
        act_batch = batch['actions'][:, :self.pred_horizon].to(self.device)
        B = act_batch.shape[0]

        img1_features = self.nets['vision_encoder1'](obs1_batch.flatten(end_dim=1))
        img1_features = img1_features.reshape(*obs1_batch.shape[:2], -1)
        img2_features = self.nets['vision_encoder2'](obs2_batch.flatten(end_dim=1))
        img2_features = img2_features.reshape(*obs2_batch.shape[:2], -1)

        features = torch.cat([img1_features, img2_features, obs_state_batch], dim=-1)
        obs_cond = features.flatten(start_dim=1)

        noise = torch.randn(act_batch.shape, device=self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device
        ).long()

        noisy_actions = self.noise_scheduler.add_noise(act_batch, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
        loss = nn.functional.mse_loss(noise_pred, noise)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        self.ema.step(self.nets.parameters())
        info['Loss'] = loss.item()

        end = time.time()
        with open('results/dp_hybrid.csv', 'a') as f:
            f.write(f"{info['Loss']},{end - start}\n")

        return info

    def get_action(self, obs_dict, goal_dict=None):
        if self.n_since_last_inference % self.act_horizon == 0:
            B = 1  # Pretend batch size to be 1
            obs1_tensor = obs_dict['agentview_image'].to(self.device)
            obs2_tensor = obs_dict['robot0_eye_in_hand_image'].to(self.device)
            state_tensor = self.process_state_obs_dict_for_evaluation(obs_dict).to(self.device)

            obs1_tensor = obs1_tensor.reshape(obs1_tensor.shape[1:])
            obs2_tensor = obs2_tensor.reshape(obs2_tensor.shape[1:])

            with torch.no_grad():
                img1_features = self.nets['vision_encoder1'](obs1_tensor)
                img2_features = self.nets['vision_encoder2'](obs2_tensor)
                obs_features = torch.cat([img1_features, img2_features, state_tensor], dim=-1)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                noisy_action = torch.randn(
                    (B, self.pred_horizon, self.act_dim), device=self.device
                )
                self.noise_scheduler.set_timesteps(self.diffusion_iters)
                for k in self.noise_scheduler.timesteps:
                    noise_pred = self.nets['noise_pred_net'](
                        sample=noisy_action, timestep=k, global_cond=obs_cond
                    )
                    noisy_action = self.noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=noisy_action
                    ).prev_sample

            self.action_pred = noisy_action.detach().to('cpu').numpy()

        # action_pred = noisy_action[:, 0].reshape((B, self.act_dim)).detach().to('cpu').numpy()
        self.n_since_last_inference = (self.n_since_last_inference + 1) % self.act_horizon
        return self.action_pred[:, self.n_since_last_inference]

    def log_info(self, info):
        log = PolicyAlgo.log_info(self, info)
        log['Loss'] = info['Loss']
        return log
