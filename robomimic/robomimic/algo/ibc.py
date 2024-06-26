"""
Implicit Behavioral Cloning for state observations.
Code based on REAL Stanford's implementation:
https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/ibc_dfo_lowdim_policy.py
"""
import torch
import torch.nn.functional as F
import time

from diffusers.optimization import get_scheduler

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.models.implicit_bc.mlp import MLP


@register_algo_factory_func('ibc')
def algo_config_to_class(algo_config):
    return IBC, {}


class IBC(PolicyAlgo):
    def _create_networks(self):
        self.epochs = 1

        self.horizon = 2
        # obs_dim: 19 for lift, 23 for can and square, 53 for tool hang
        self.obs_dim = 0
        for k in list(self.obs_key_shapes.keys()):
            self.obs_dim += self.obs_key_shapes[k][0]
        self.act_dim = 7
        self.n_obs_steps = 2
        self.n_act_steps = 1
        self.train_n_neg = 8
        self.pred_n_samples = 8
        self.pred_n_iter = 5

        self.model = MLP(  # IbcDfoLowdimPolicy(
            self.horizon, self.obs_dim, self.act_dim, self.n_act_steps, self.n_obs_steps,
            train_n_neg=self.train_n_neg, pred_n_samples=self.pred_n_samples
        )
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=1e-4, weight_decay=1e-6
        )
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=500 * self.epochs  # len(dataloader) * epochs
        )
        # Clear file before write-up
        with open('results/ibc.csv', 'w+') as f:
            f.write('loss,time\n')

    @staticmethod
    def process_obs_dict_for_training(obs_dict):
        obs_list = list()
        for k in list(obs_dict.keys()):
            obs_list.append(obs_dict[k])

        obs_tensor = torch.cat(obs_list, dim=2).to(torch.float32)
        return obs_tensor

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

        # obs_tensor = torch.zeros((1, self.obs_horizon, self.obs_dim), dtype=torch.float32)
        # # The last dim of obs_list is 3 + 4 + 2 = 9
        # obs_tensor[:, :, :9] = torch.cat(obs_list, dim=2)
        obs_tensor = torch.cat(obs_list, dim=1)
        obs_tensor = obs_tensor.reshape(obs_tensor.shape[1], 1, obs_tensor.shape[2])
        return obs_tensor

    @staticmethod
    def get_action_stats(act_batch):
        act_batch = act_batch.reshape(-1, act_batch.shape[-1])
        stats = {
            'min': torch.min(act_batch),
            'max': torch.max(act_batch)
        }
        return stats

    def train_on_batch(self, batch, epoch, validate=False):
        start = time.time()
        info = super(IBC, self).train_on_batch(batch, epoch, validate=validate)

        obs_batch = self.process_obs_dict_for_training(batch['obs']).to(self.device)
        act_batch = batch['actions'].to(self.device)
        B = act_batch.shape[0]

        this_obs = obs_batch[:, :self.n_obs_steps]
        start = self.n_obs_steps - 1
        end = start + self.n_act_steps
        this_act = act_batch[:, start:end]

        # Small additive noise to true positives.
        this_act += torch.normal(
            mean=0, std=1e-4, size=this_act.shape, dtype=this_act.dtype, device=this_act.device
        )
        # Sample negatives: (B, train_n_neg, n_act_steps, n_obs_steps)
        act_stats = self.get_action_stats(act_batch)
        act_dist = torch.distributions.Uniform(low=act_stats['min'], high=act_stats['max'])
        samples = act_dist.sample((B, self.train_n_neg, self.n_act_steps, self.act_dim)).to(dtype=this_act.dtype)
        # this_act = this_act.reshape((B, n_obs_steps))
        # samples = samples.reshape((B, train_n_neg, n_obs_steps))

        act_samples = torch.cat([this_act.unsqueeze(dim=1), samples], dim=1)
        act_samples = act_samples.reshape((B, self.train_n_neg + 1, self.n_act_steps, self.act_dim))
        # (B, train_n_neg+1, n_act_steps, n_obs_steps)

        labels = torch.zeros((B,), dtype=torch.int64, device=this_act.device)
        logits = self.model.forward(this_obs, act_samples)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        info['Loss'] = loss.item()

        end = time.time()
        with open('results/ibc.csv', 'a') as f:
            f.write(f"{info['Loss']},{end - start}\n")

        return info

    def get_action(self, obs_dict, goal_dict=None):
        obs_tensor = self.process_obs_dict_for_evaluation(obs_dict)
        # act_tensor = obs_dict['actions']
        B = obs_tensor.shape[0]

        this_obs = obs_tensor[:, :self.n_obs_steps]
        # act_stats = self.get_action_stats(act)
        act_stats = {'min': 0, 'max': 1}

        act_dist = torch.distributions.Uniform(
            low=act_stats['min'],
            high=act_stats['max']
        )
        samples = act_dist.sample((B, self.pred_n_samples, self.n_act_steps, self.act_dim)).to(dtype=this_obs.dtype)
        # (B, N, Ta, Da)

        print(this_obs.shape)
        print(samples.shape)

        # Set 1 as B (batch size) to satisfy model.forward()
        this_obs = this_obs.reshape((1, self.n_obs_steps, self.obs_dim))
        samples = samples.reshape((1,  self.n_obs_steps, self.pred_n_samples, self.obs_dim))

        noise_scale = 3e-2
        with torch.no_grad():
            for _ in range(self.pred_n_iter):
                logits = self.model.forward(this_obs, samples)
                probs = F.softmax(logits, dim=-1)

                # Resample with replacement
                indices = torch.multinomial(probs, self.pred_n_samples, replacement=True)
                samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), indices]

                # Add noise and clip to target bounds
                samples += torch.randn_like(samples) * noise_scale
                samples = samples.clamp(min=act_stats['min'], max=act_stats['max'])

            # Return target with highest probability
            logits = self.model.forward(this_obs, samples)
            probs = F.softmax(logits, dim=-1)
            best_idxs = probs.argmax(dim=-1)
            acts = samples[torch.arange(samples.size(0)), best_idxs, :]

        return acts

    def log_info(self, info):
        log = PolicyAlgo.log_info(self, info)
        log['Loss'] = info['Loss']
        return log
