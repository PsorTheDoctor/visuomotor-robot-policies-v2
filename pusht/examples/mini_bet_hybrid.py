"""
Behavior Transformer for hybrid observations.
Code based on miniBET implementation:
https://github.com/notmahi/miniBET
"""
import numpy as np
import torch
import time
from skvideo.io import vwrite
from tqdm.auto import tqdm

from utils.load_data import *
from utils.dataset import normalize_data, unnormalize_data
from utils.resnet import *
from behavior_transformer import BehaviorTransformer, GPT, GPTConfig

batch_size = 64
horizon = 3

env, dataset, stats, dataloader = load_pusht_image_data_for_training(
    pred_horizon=horizon, obs_horizon=horizon, act_horizon=horizon
)

"""
Network
"""
conditional = False
vision_feature_dim = 512
lowdim_obs_dim = 2
obs_dim = vision_feature_dim + lowdim_obs_dim
act_dim = 2
goal_dim = obs_dim if conditional else 0
device = torch.device('cuda')

vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)
vision_encoder = vision_encoder.to(device)

bet = BehaviorTransformer(
    obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim,
    gpt_model=GPT(GPTConfig(
        block_size=3, input_dim=obs_dim, n_layer=4, n_head=4, n_embd=72
    )),
    n_clusters=24, kmeans_fit_steps=50
).to(device)

optimizer = bet.configure_optimizers(
    weight_decay=0.1, learning_rate=1e-4, betas=[0.9, 0.999]
)

"""
Training
"""
epochs = 1000

start = time.time()
with tqdm(range(epochs), desc='Epoch') as tglobal:
    total_loss = list()
    for epoch_idx in tglobal:
        epoch_loss = list()
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for batch in tepoch:
                nimage = batch['image'].to(device)
                nagent_pos = batch['agent_pos'].to(device)
                naction = batch['action'].to(device)
                B = nagent_pos.shape[0]

                img_features = vision_encoder(nimage.flatten(end_dim=1))
                img_features = img_features.reshape(*nimage.shape[:2], -1)

                obs_features = torch.cat([img_features, nagent_pos], dim=-1)

                optimizer.zero_grad()
                _, loss, loss_dict = bet(obs_features, None, naction)
                loss.backward()
                optimizer.step()

                epoch_loss.append(float(loss))
                tepoch.set_postfix(loss=loss)

        tglobal.set_postfix(loss=np.mean(epoch_loss))
        total_loss.append(np.mean(epoch_loss))


end = time.time()
minutes, seconds = divmod(end - start, 60)
with open('../results/mini_bet_hybrid.txt', 'w+') as f:
    f.write(f'Training took {round(minutes)}m {round(seconds)}s./n')

np.save('../results/mini_bet_hybrid.npy', total_loss)


def inference():
    max_steps = 300
    env = PushTImageEnv()
    env.seed(100000)  # use a seed >200 to avoid initial states seen in the training data

    obs, info = env.reset()
    obs_deque = collections.deque([obs] * horizon, maxlen=horizon)
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc='Eval PushImageEnv') as pbar:
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

                act, _, _ = bet(obs_features, None, None)

            act = act.detach().to('cpu').numpy()[0]
            act_pred = unnormalize_data(act, stats=stats['action'])

            start = horizon - 1
            end = start + horizon
            act = act_pred[start:end, :]

            # Execute action_horizon number of steps without replanning
            for i in range(len(act)):
                obs, reward, done, _, info = env.step(act[i])
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
    with open('../results/mini_bet_hybrid.txt', 'a') as f:
        f.write(f'Score: {max(rewards)}')

    return imgs, max(rewards)


rewards = []
for _ in range(10):
    imgs, reward = inference()
    rewards.append(reward)

with open('../results/mini_bet_hybrid.txt', 'a') as f:
    f.write(f'Mean: {np.mean(rewards)}\n')
    f.write(f'Std dev: {np.std(rewards)}\n')

vwrite('../results/mini_bet_hybrid.mp4', imgs)
