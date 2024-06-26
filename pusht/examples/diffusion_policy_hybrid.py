"""
Diffusion Policy for hybrid observations.
Code based on the official implementation:
https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing
"""
import numpy as np
import torch
import collections
import time
from skvideo.io import vwrite

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from utils.env import PushTImageEnv
from utils.dataset import PushTImageDataset, normalize_data, unnormalize_data
from diffusion_policy.unet import ConditionalUnet1D
from utils.resnet import *

env = PushTImageEnv()
env.seed(1000)
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

dataset_path = '../data/pusht_cchi_v7_replay.zarr.zip'

pred_horizon = 16
obs_horizon = 2
action_horizon = 8

dataset = PushTImageDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
stats = dataset.stats

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    num_workers=1,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)
batch = next(iter(dataloader))
print("batch['image'].shape:", batch['image'].shape)
print("batch['agent_pos'].shape:", batch['agent_pos'].shape)
print("batch['action'].shape", batch['action'].shape)

"""
Network
"""
vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)
vision_feature_dim = 512
lowdim_obs_dim = 2
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 2
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})
diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)
device = torch.device('cuda')
_ = nets.to(device)


def inference():
    max_steps = 300
    env = PushTImageEnv()
    env.seed(100000)  # use a seed >200 to avoid initial states seen in the training data

    obs, info = env.reset()
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc='Eval PushTImageEnv') as pbar:
        while not done:
            B = 1
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            agent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            images = torch.from_numpy(images).to(device, dtype=torch.float32)
            agent_poses = torch.from_numpy(agent_poses).to(device, dtype=torch.float32)

            with torch.no_grad():
                img_features = ema_nets['vision_encoder'](images)
                obs_features = torch.cat([img_features, agent_poses], dim=-1)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # Initialize action from Gaussian noise
                noisy_act = torch.randn(
                    (B, pred_horizon, action_dim), device=device
                )
                act = noisy_act
                noise_scheduler.set_timesteps(diffusion_iters)
                for k in noise_scheduler.timesteps:
                    # Predict noise
                    noise_pred = ema_nets['noise_pred_net'](
                        sample=act, timestep=k, global_cond=obs_cond
                    )
                    # Inverse diffusion step (remove noise)
                    act = noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=act
                    ).prev_sample

            act = act.detach().to('cpu').numpy()
            act = act[0]
            act_pred = unnormalize_data(act, stats=stats['action'])

            start = obs_horizon - 1
            end = start + action_horizon
            action = act_pred[start:end, :]

            # Execute action_horizon number of steps without replanning
            for i in range(len(action)):
                obs, reward, done, _, info = env.step(action[i])
                obs_deque.append(obs)
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps or reward == 1.0:
                    done = True
                if done:
                    break

    print('Score:', max(rewards))
    with open('../results_locked/dp_hybrid.txt', 'a') as f:
        f.write(f'Epoch: {epoch_idx + 1} Score: {max(rewards)}\n')

    return imgs


"""
Training
"""
epochs = 100

# Exponential Moving Average
ema = EMAModel(parameters=nets.parameters(), power=0.75)

optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6
)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * epochs
)

imgs = None
start = time.time()
with tqdm(range(epochs), desc='Epoch') as tglobal:
    total_loss = list()
    for epoch_idx in tglobal:
        epoch_loss = list()
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                nimage = nbatch['image'][:, :obs_horizon].to(device)
                nagent_pos = nbatch['agent_pos'][:, :obs_horizon].to(device)
                naction = nbatch['action'].to(device)
                B = nagent_pos.shape[0]

                img_features = nets['vision_encoder'](nimage.flatten(end_dim=1))
                img_features = img_features.reshape(*nimage.shape[:2], -1)

                obs_features = torch.cat([img_features, nagent_pos], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)

                noise = torch.randn(naction.shape, device=device)

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps
                )
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond
                )
                loss = nn.functional.mse_loss(noise_pred, noise)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                ema.step(nets.parameters())

                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

        tglobal.set_postfix(loss=np.mean(epoch_loss))
        total_loss.append(np.mean(epoch_loss))

ema_nets = nets
ema.copy_to(ema_nets.parameters())

end = time.time()
minutes, seconds = divmod(end - start, 60)
with open('../results_locked/dp_hybrid.txt', 'a') as f:
    f.write(f'Training took {round(minutes)}m {round(seconds)}s.\n')

for _ in range(10):
    imgs = inference()

np.save('../results_locked/dp_hybrid.npy', total_loss)

vwrite('../results/dp_hybrid.mp4', imgs)
