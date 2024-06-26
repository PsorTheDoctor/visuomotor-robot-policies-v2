import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, horizon, obs_dim, action_dim, n_action_steps, n_obs_steps,
                 dropout=0.1, train_n_neg=1024, pred_n_iter=5, pred_n_samples=1024):
        super().__init__()

        in_action_channels = action_dim * n_action_steps  # 2 * 1 = 2
        in_obs_channels = obs_dim * n_obs_steps  # 5 * 2 = 10
        in_channels = in_action_channels + in_obs_channels  # 2 + 10 = 12
        mid_channels = 1024
        out_channels = 1

        self.dense0 = nn.Linear(in_features=in_channels, out_features=mid_channels)
        self.drop0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop3 = nn.Dropout(dropout)
        self.dense4 = nn.Linear(in_features=mid_channels, out_features=out_channels)

        # self.normalizer = LinearNormalizer()

        self.train_n_neg = train_n_neg
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon

    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape
        s = obs.reshape(B, 1, -1).expand(-1, N, -1)
        x = torch.cat([s, action.reshape(B, N, -1)], dim=-1).reshape(B * N, -1)

        x = self.drop0(torch.relu(self.dense0(x)))
        x = self.drop1(torch.relu(self.dense1(x)))
        x = self.drop2(torch.relu(self.dense2(x)))
        x = self.drop3(torch.relu(self.dense3(x)))
        x = self.dense4(x)
        x = x.reshape(B, N)
        return x
