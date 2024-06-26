"""
Implicit Behavioral Cloning for hybrid observations.
Training code based on REAL Stanford's implementation:
https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/ibc_dfo_lowdim_policy.py
"""
import torch.nn.functional as F
import time
from skvideo.io import vwrite
from tqdm.auto import tqdm

from diffusers.optimization import get_scheduler

from utils.load_data import *
from utils.dataset import normalize_data, unnormalize_data
from utils.resnet import *
from implicit_bc.mlp import MLP

device = torch.device('cuda')
horizon = 2
vision_feature_dim = 512
lowdim_obs_dim = 2
obs_dim = vision_feature_dim + lowdim_obs_dim
act_dim = 2
n_obs_steps = 2
n_act_steps = 1
train_n_neg = 8
pred_n_samples = 512
pred_n_iter = 5
epochs = 1

vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)
vision_encoder = vision_encoder.to(device)

env, dataset, stats, dataloader = load_pusht_image_data_for_training(
    pred_horizon=horizon, obs_horizon=horizon, act_horizon=horizon
)
model = MLP(
    horizon, obs_dim, act_dim, n_act_steps, n_obs_steps,
    train_n_neg=train_n_neg, pred_n_samples=pred_n_samples
).to(device)

optimizer = torch.optim.AdamW(
    params=model.parameters(), lr=1e-4, weight_decay=1e-6  # weight_decay=2e-4, lr=1e-5
)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * epochs
)
# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(params)

def get_action_stats(act):
    act = act.reshape(-1, act.shape[-1])
    stats = {
        'min': torch.min(act),
        'max': torch.max(act)
    }
    return stats


def compute_loss(obs, act):
    # Input is already normalized
    obs = obs.to(device)
    act = act.to(device)
    B = act.shape[0]

    this_obs = obs[:, :n_obs_steps]
    start = n_obs_steps - 1
    end = start + n_act_steps
    this_act = act[:, start:end]

    # Small additive noise to true positives.
    this_act += torch.normal(
        mean=0, std=1e-4, size=this_act.shape, dtype=this_act.dtype, device=this_act.device
    )
    # Sample negatives: (B, train_n_neg, n_act_steps, n_obs_steps)
    act_stats = get_action_stats(act)
    act_dist = torch.distributions.Uniform(low=act_stats['min'], high=act_stats['max'])
    samples = act_dist.sample((B, train_n_neg, n_act_steps, n_obs_steps)).to(dtype=this_act.dtype)
    # this_act = this_act.reshape((B, n_obs_steps))
    # samples = samples.reshape((B, train_n_neg, n_obs_steps))

    act_samples = torch.cat([this_act.unsqueeze(dim=1), samples], dim=1)
    act_samples = act_samples.reshape((B, train_n_neg + 1, n_act_steps, n_obs_steps))
    # (B, train_n_neg+1, n_act_steps, n_obs_steps)

    # Get onehot labels
    labels = torch.zeros(act_samples.shape[:2], dtype=this_act.dtype, device=this_act.device)
    labels[:, 0] = 1
    logits = model.forward(this_obs, act_samples)
    # (B, N)
    logits = torch.log_softmax(logits, dim=-1)
    loss = -torch.mean(torch.sum(logits * labels, axis=-1))
    return loss


def predict_action(obs):
    B = 1
    obs = obs.reshape((B, obs.shape[1], obs.shape[2]))
    this_obs = obs[:, :n_obs_steps].to(dtype=torch.float32).to(device)
    # act_stats = get_action_stats(act)
    act_stats = {
        'min': torch.tensor(-0.99).to(device),
        'max': torch.tensor(0.99).to(device)
    }
    action_dist = torch.distributions.Uniform(
        low=act_stats['min'],
        high=act_stats['max']
    )
    samples = action_dist.sample((B, pred_n_samples, n_act_steps, act_dim)).to(dtype=torch.float32).to(device)
    # (B, N, Ta, Da)

    noise_scale = 3e-2
    for i in range(pred_n_iter):
        # Compute energies.
        logits = model.forward(this_obs, samples)
        probs = F.softmax(logits, dim=-1)

        # Resample with replacement.
        idxs = torch.multinomial(probs, pred_n_samples, replacement=True)
        samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

        # Add noise and clip to target bounds.
        samples = samples + torch.randn_like(samples) * noise_scale
        samples = samples.clamp(min=act_stats['min'], max=act_stats['max'])

        # Return target with highest probability.
        logits = model.forward(this_obs, samples)
        probs = F.softmax(logits, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        acts_n = samples[torch.arange(samples.size(0)), best_idxs, :]

    return acts_n


def inference(verbose=True):
    max_steps = 300
    env = PushTImageEnv()
    env.seed(100000)  # use a seed >200 to avoid initial states seen in the training data

    obs, info = env.reset()
    obs_deque = collections.deque([obs] * horizon, maxlen=horizon)
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc='Eval PushTStateEnv') as pbar:
        while not done:
            B = 1
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            agent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            images = torch.from_numpy(images).to(device, dtype=torch.float32)
            agent_poses = torch.from_numpy(agent_poses).to(device, dtype=torch.float32)

            with torch.no_grad():
                img_features = vision_encoder(images)
                obs_features = torch.cat([img_features, agent_poses], dim=-1)
                obs_features = obs_features.reshape((B, horizon, obs_dim))

                action = predict_action(obs_features)

            action = action.detach().to('cpu').numpy()
            action = action[0]
            action = unnormalize_data(action, stats=stats['action'])

            # start = horizon - 1
            # end = start + horizon
            # action = action_pred[start:end, :]

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

    # if verbose:
    #     print('Score:', max(rewards))
    #     with open('../results/ibc_hybrid_4.txt', 'a') as f:
    #         f.write(f'Score: {max(rewards)}\n')

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
            for batch in tepoch:
                image = batch['image'].to(device)
                agent_pos = batch['agent_pos'].to(device)
                action = batch['action'].to(device)
                B = agent_pos.shape[0]

                img_features = vision_encoder(image.flatten(end_dim=1))
                img_features = img_features.reshape(*image.shape[:2], -1)

                obs_features = torch.cat([img_features, agent_pos], dim=-1)

                loss = compute_loss(obs_features, action)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

        tglobal.set_postfix(loss=np.mean(epoch_loss))
        total_loss.append(np.mean(epoch_loss))

        if (epoch_idx + 1) % 100 == 0:
            rewards = []
            for _ in range(20):
                imgs, reward = inference(verbose=False)
                rewards.append(reward)

            success_rate = np.sum(np.array(rewards) > 0.8)
            # with open('../results/ibc_hybrid_4.txt', 'a') as f:
            #     f.write(f'Epoch: {epoch_idx + 1} Score: {success_rate}\n')

end = time.time()
minutes, seconds = divmod(end - start, 60)
# with open('../results/ibc_hybrid_4.txt', 'a') as f:
#     f.write(f'Training took {round(minutes)}m {round(seconds)}s.')
#
# np.save('../results/ibc_hybrid_4.npy', total_loss)

start = time.time()
rewards = []
for _ in range(1):
    imgs, reward = inference()
    rewards.append(reward)
end = time.time()
minutes, seconds = divmod(end - start, 60)
print(f'{minutes}:{seconds}')


# with open('../results/ibc_hybrid_4.txt', 'a') as f:
#     f.write(f'Mean: {np.mean(rewards)}\n')
#     f.write(f'Std dev: {np.std(rewards)}\n')
#
# vwrite('../results/ibc_hybrid_4.mp4', imgs)
