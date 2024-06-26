"""
Behavior Transformer for visual observations only.
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

batch_size = 256
horizon = 3

env, dataset, stats, dataloader = load_pusht_image_data_for_training(
    pred_horizon=horizon, obs_horizon=horizon, act_horizon=horizon
)

"""
Network
"""
conditional = False
obs_dim = 512
act_dim = 2
goal_dim = obs_dim if conditional else 0
device = torch.device('cuda')

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
vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)
vision_encoder = vision_encoder.to(device)


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

    with tqdm(total=max_steps, desc='Eval PushTImageEnv') as pbar:
        while not done:
            B = 1
            images = np.stack([x['image'] for x in obs_deque])
            images = torch.from_numpy(images).to(device, dtype=torch.float32)

            with torch.no_grad():
                obs = vision_encoder(images)
                obs = obs.reshape((B, horizon, obs_dim))

                with torch.no_grad():
                    act, _, _ = bet(obs, None, None)

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
    with open('../results_locked/mini_bet_vision.txt', 'a') as f:
        f.write(f'Score: {max(rewards)}\n')

    return imgs, max(rewards)


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
                obs = batch['image'].to(device)
                act = batch['action'].to(device)
                B = obs.shape[0]

                features = vision_encoder(obs.flatten(end_dim=1))
                features = features.reshape(*obs.shape[:2], -1)

                optimizer.zero_grad()
                _, loss, loss_dict = bet(features, None, act)
                loss.backward()
                optimizer.step()

                epoch_loss.append(float(loss))
                tepoch.set_postfix(loss=loss)

        tglobal.set_postfix(loss=np.mean(epoch_loss))
        total_loss.append(np.mean(epoch_loss))


end = time.time()
minutes, seconds = divmod(end - start, 60)
with open('../results_locked/mini_bet_vision.txt', 'w+') as f:
    f.write(f'Training took {round(minutes)}m {round(seconds)}s.\n')

rewards = []
for _ in range(10):
    imgs, reward = inference()
    rewards.append(reward)

with open('../results_locked/mini_bet_vision.txt', 'a') as f:
    f.write(f'Mean: {np.mean(rewards)}\n')
    f.write(f'Std dev: {np.std(rewards)}\n')

np.save('../results_locked/mini_bet_vision.npy', total_loss)

vwrite('../results/mini_bet_vision.mp4', imgs)
