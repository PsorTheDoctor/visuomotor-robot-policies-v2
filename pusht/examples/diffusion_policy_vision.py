"""
Diffusion Policy for visual observations only.
Code based on the official implementation:
https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing
"""
import time
from skvideo.io import vwrite
from tqdm.auto import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from utils.load_data import *
from utils.dataset import unnormalize_data
from diffusion_policy.unet import ConditionalUnet1D
from utils.resnet import *

pred_horizon = 16
obs_horizon = 2
act_horizon = 8

env, dataset, stats, dataloader = load_pusht_image_data_for_training(
    pred_horizon=pred_horizon, obs_horizon=obs_horizon, act_horizon=act_horizon
)

"""
Network
"""
vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)
obs_dim = 512  # vision feature dim
action_dim = 2
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon
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

# params = sum(p.numel() for p in vision_encoder.parameters() if p.requires_grad)
# print(params)


"""
Training
"""
epochs = 100

# Exponential Moving Average
ema = EMAModel(parameters=nets.parameters(), power=0.75)

optimizer = torch.optim.AdamW(
    params=nets.parameters(), lr=1e-4, weight_decay=1e-6
)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * epochs
)


def inference(_nets, _ema, verbose=True):
    nets = _nets
    _this_ema = _ema
    _this_ema.copy_to(nets.parameters())

    max_steps = 300
    env = PushTImageEnv()
    env.seed(100000)  # use a seed >200 to avoid initial states seen in the training data
    obs, info = env.reset()
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon
    )
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc='Eval PushTImageEnv') as pbar:
        while not done:
            B = 1
            images = np.stack([x['image'] for x in obs_deque])
            images = torch.from_numpy(images).to(device, dtype=torch.float32)

            with torch.no_grad():
                obs_features = nets['vision_encoder'](images)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # Initialize action from Gaussian noise
                noisy_act = torch.randn(
                    (B, pred_horizon, action_dim), device=device
                )
                act = noisy_act
                noise_scheduler.set_timesteps(diffusion_iters)
                for k in noise_scheduler.timesteps:
                    # Predict noise
                    noise_pred = nets['noise_pred_net'](
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
            end = start + act_horizon
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

    if verbose:
        print('Score:', max(rewards))
        with open('../results/dp_vision_3.txt', 'a') as f:
            f.write(f'Epoch: {epoch_idx} Score: {max(rewards)}\n')

    return imgs, max(rewards)


start = time.time()
with tqdm(range(epochs), desc='Epoch') as tglobal:
    total_loss = list()
    for epoch_idx in tglobal:
        epoch_loss = list()
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for batch in tepoch:
                obs = batch['image'][:, :obs_horizon].to(device)
                act = batch['action'].to(device)
                B = obs.shape[0]

                features = nets['vision_encoder'](obs.flatten(end_dim=1))
                features = features.reshape(*obs.shape[:2], -1)
                obs_cond = features.flatten(start_dim=1)

                noise = torch.randn(act.shape, device=device)

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                noisy_actions = noise_scheduler.add_noise(
                    act, noise, timesteps
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

        if (epoch_idx + 1) % 10 == 0:
            rewards = []
            for _ in range(20):
                imgs, reward = inference(nets, ema, verbose=False)
                rewards.append(reward)

            success_rate = np.sum(np.array(rewards) > 0.8)
            with open('../results/dp_vision_3.txt', 'a') as f:
                f.write(f'Epoch: {epoch_idx + 1} Score: {success_rate}\n')

ema_nets = nets
ema.copy_to(ema_nets.parameters())

end = time.time()
minutes, seconds = divmod(end - start, 60)
with open('../results/dp_vision_3.txt', 'a') as f:
    f.write(f'Training took {round(minutes)}m {round(seconds)}s.\n')

np.save('../results/dp_vision_3.npy', total_loss)

rewards = []
for _ in range(10):
    imgs, reward = inference(nets, ema)
    rewards.append(reward)

with open('../results/dp_vision_3.txt', 'a') as f:
    f.write(f'Mean: {np.mean(rewards)}\n')
    f.write(f'Std dev: {np.std(rewards)}\n')

vwrite('../results/dp_vision_3.mp4', imgs)
