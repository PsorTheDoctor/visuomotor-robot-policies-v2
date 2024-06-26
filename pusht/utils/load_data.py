import torch
import collections
import numpy as np

from utils.env import PushTEnv, PushTImageEnv
from utils.dataset import PushTStateDataset, PushTImageDataset

DATASET_PATH = '../data/pusht_cchi_v7_replay.zarr.zip'


def load_pusht_state_data_for_training(
    batch_size=256, pred_horizon=16, obs_horizon=2, act_horizon=8, verbose=True
):
    env = PushTEnv()
    env.seed(1000)
    obs, IGNORE_GIT_FOLDER_PATTERNS = env.reset()
    act = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(act)

    dataset = PushTStateDataset(
        dataset_path=DATASET_PATH,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=act_horizon
    )
    stats = dataset.stats

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,  # batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    if verbose:
        batch = next(iter(dataloader))
        print("batch['obs'].shape:", batch['obs'].shape)
        print("batch['action'].shape", batch['action'].shape)

    # print(batch['obs'][0])
    # print(batch['action'][0])
    return env, dataset, stats, dataloader


def load_pusht_image_data_for_training(
    batch_size=64, pred_horizon=16, obs_horizon=2, act_horizon=8, verbose=True
):
    env = PushTImageEnv()
    env.seed(1000)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    dataset = PushTImageDataset(
        dataset_path=DATASET_PATH,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=act_horizon
    )
    stats = dataset.stats

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    if verbose:
        batch = next(iter(dataloader))
        print("batch['image'].shape:", batch['image'].shape)
        print("batch['agent_pos'].shape:", batch['agent_pos'].shape)
        print("batch['action'].shape", batch['action'].shape)

    return env, dataset, stats, dataloader


# _, _, stats,dataloader = load_pusht_state_data_for_training()
# batch = next(iter(dataloader))
# print("batch['agent_pos'].shape:", batch['agent_pos'])
# print("batch['action'].shape", batch['action'])
