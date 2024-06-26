"""
Diffusion Policy for state observations.
Code based on the official implementation:
https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing
"""
import numpy as np
import torch
import torch.nn as nn
import collections
import time
from skvideo.io import vwrite

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from utils.env import PushTEnv
from utils.dataset import PushTStateDataset, normalize_data, unnormalize_data
from diffusion_policy.unet import ConditionalUnet1D

env = PushTEnv()
env.seed(1000)
obs, IGNORE_GIT_FOLDER_PATTERNS = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

dataset_path = '../data/pusht_cchi_v7_replay.zarr.zip'

pred_horizon = 16
obs_horizon = 2
action_horizon = 8

dataset = PushTStateDataset(
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
print("batch['obs'].shape:", batch['obs'].shape)
print("batch['action'].shape", batch['action'].shape)

obs_dim = 5
action_dim = 2

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)
diffusion_iters = 100
diffusion_iters_eval = 16

noise_scheduler = DDPMScheduler(
    num_train_timesteps=diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)
device = torch.device('cuda')
_ = noise_pred_net.to(device)

epochs = 100

# Exponential Moving Average
ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)

optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6
)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * epochs
)
# params = sum(p.numel() for p in noise_pred_net.parameters() if p.requires_grad)
# print(params)


def inference(_noise_pred_net, _ema, verbose=True):
    ema_noise_pred_net = _noise_pred_net
    _this_ema = _ema
    _this_ema.copy_to(ema_noise_pred_net.parameters())

    max_steps = 300
    env = PushTEnv()
    env.seed(100000)  # use a seed >200 to avoid initial states seen in the training data

    obs, info = env.reset()
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc='Eval PushTStateEnv') as pbar:
        while not done:
            B = 1
            obs_seq = np.stack(obs_deque)
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            with torch.no_grad():
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                # Initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device
                )
                naction = noisy_action
                noise_scheduler.set_timesteps(diffusion_iters_eval)
                for k in noise_scheduler.timesteps:
                    # Predict noise
                    noise_pred = ema_noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # Inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            naction = naction.detach().to('cpu').numpy()
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end, :]

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

    env.save_action_history('../results/dp_kan_history.npy')

    if verbose:
        print('Score:', max(rewards))
        with open('../results/dp_kan.txt', 'a') as f:
            f.write(f'Score: {max(rewards)}\n')

    return imgs, max(rewards)


"""
Training
"""
start = time.time()
with tqdm(range(epochs), desc='Epoch') as tglobal:
    total_loss = list()
    for epoch_idx in tglobal:
        epoch_loss = list()
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                nobs = nbatch['obs'].to(device)
                naction = nbatch['action'].to(device)
                B = nobs.shape[0]

                obs_cond = nobs[:, :obs_horizon, :]
                obs_cond = obs_cond.flatten(start_dim=1)

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
                ema.step(noise_pred_net.parameters())

                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

        tglobal.set_postfix(loss=np.mean(epoch_loss))
        total_loss.append(np.mean(epoch_loss))

        if (epoch_idx + 1) % 10 == 0:
            rewards = []
            for _ in range(20):
                imgs, reward = inference(noise_pred_net, ema, verbose=False)
                rewards.append(reward)

            success_rate = np.sum(np.array(rewards) > 0.8)
            with open('../results/dp_kan.txt', 'a') as f:
                f.write(f'Epoch: {epoch_idx + 1} Score: {success_rate}\n')


ema_noise_pred_net = noise_pred_net
ema.copy_to(ema_noise_pred_net.parameters())

end = time.time()
minutes, seconds = divmod(end - start, 60)
with open('../results/dp_kan.txt', 'a') as f:
    f.write(f'Training took {round(minutes)}m {round(seconds)}s.\n')

np.save('../results/dp_kan.npy', total_loss)

rewards = []
for _ in range(10):
    imgs, reward = inference(noise_pred_net, ema)
    rewards.append(reward)

with open('../results/dp_kan.txt', 'a') as f:
    f.write(f'Mean: {np.mean(rewards)}\n')
    f.write(f'Std dev: {np.std(rewards)}\n')

vwrite('../results/dp_kan.mp4', imgs)
