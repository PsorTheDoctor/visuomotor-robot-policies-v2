"""
Behavior Transformer for state observations.
Code based on miniBET implementation:
https://github.com/notmahi/miniBET
"""
import numpy as np
import torch
import time
from skvideo.io import vwrite
from tqdm.auto import tqdm

from diffusers.optimization import get_scheduler

from utils.load_data import *
from utils.dataset import normalize_data, unnormalize_data
from behavior_transformer import BehaviorTransformer, GPT, GPTConfig

batch_size = 256
horizon = 3

env, dataset, stats, dataloader = load_pusht_state_data_for_training(
    pred_horizon=horizon, obs_horizon=horizon, act_horizon=horizon
)

"""
Network
"""
conditional = False
obs_dim = 5
act_dim = 2
goal_dim = obs_dim if conditional else 0
device = torch.device('cuda')
epochs = 1000

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
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * epochs
)
params = sum(p.numel() for p in bet.parameters() if p.requires_grad)
print(params)

# def inference(verbose=True):
#     max_steps = 300
#     env = PushTEnv()
#     env.seed(100000)  # use a seed >200 to avoid initial states seen in the training data
#
#     obs, info = env.reset()
#     obs_deque = collections.deque([obs] * horizon, maxlen=horizon)
#     imgs = [env.render(mode='rgb_array')]
#     rewards = list()
#     done = False
#     step_idx = 0
#
#     with tqdm(total=max_steps, desc='Eval PushTStateEnv') as pbar:
#         while not done:
#             B = 1
#             obs = np.stack(obs_deque)
#             obs = normalize_data(obs, stats=stats['obs'])
#             obs = torch.from_numpy(obs).to(device, dtype=torch.float32)
#
#             obs = obs.reshape((B, horizon, obs_dim))
#             with torch.no_grad():
#                 act, _, _ = bet(obs, None, None)
#
#             act = act.detach().to('cpu').numpy()[0]
#             act_pred = unnormalize_data(act, stats=stats['action'])
#
#             start = horizon - 1
#             end = start + horizon
#             act = act_pred[start:end, :]
#
#             # Execute action_horizon number of steps without replanning
#             for i in range(len(act)):
#                 obs, reward, done, _, info = env.step(act[i])
#                 obs_deque.append(obs)
#                 rewards.append(reward)
#                 imgs.append(env.render(mode='rgb_array'))
#
#                 step_idx += 1
#                 pbar.update(1)
#                 pbar.set_postfix(reward=reward)
#                 if step_idx > max_steps or reward == 1.0:
#                     done = True
#                 if done:
#                     break
#
#     if verbose:
#         print('Score:', max(rewards))
#         with open('../results/mini_bet_state.txt', 'a') as f:
#             f.write(f'Score: {max(rewards)}\n')
#
#     return imgs, max(rewards)
#
#
# """
# Training
# """
# start = time.time()
# with tqdm(range(epochs), desc='Epoch') as tglobal:
#     total_loss = list()
#     for epoch_idx in tglobal:
#         epoch_loss = list()
#         with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
#             for batch in tepoch:
#                 obs_seq = batch['obs'].to(device)
#                 act_seq = batch['action'].to(device)
#
#                 optimizer.zero_grad()
#                 _, loss, loss_dict = bet(obs_seq, None, act_seq)
#                 loss.backward()
#                 optimizer.step()
#                 lr_scheduler.step()
#
#                 loss_cpu = loss.item()
#                 epoch_loss.append(loss_cpu)
#                 tepoch.set_postfix(loss=loss_cpu)
#
#         tglobal.set_postfix(loss=np.mean(epoch_loss))
#         total_loss.append(np.mean(epoch_loss))
#
#         if (epoch_idx + 1) % 100 == 0:
#             rewards = []
#             for _ in range(20):
#                 imgs, reward = inference(verbose=False)
#                 rewards.append(reward)
#
#             success_rate = np.sum(np.array(rewards) > 0.8)
#             with open('../results/mini_bet_state.txt', 'a') as f:
#                 f.write(f'Epoch: {epoch_idx + 1} Score: {success_rate}\n')
#
# end = time.time()
# minutes, seconds = divmod(end - start, 60)
# with open('../results/mini_bet_state.txt', 'a') as f:
#     f.write(f'Training took {round(minutes)}m {round(seconds)}s.\n')
#
# np.save('../results/mini_bet_state.npy', total_loss)
#
# rewards = []
# for _ in range(10):
#     imgs, reward = inference()
#     rewards.append(reward)
#
# with open('../results/mini_bet_state.txt', 'a') as f:
#     f.write(f'Mean: {np.mean(rewards)}\n')
#     f.write(f'Std dev: {np.std(rewards)}\n')
#
# vwrite('../results/mini_bet.mp4', imgs)
