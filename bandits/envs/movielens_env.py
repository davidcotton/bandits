import numpy as np
from typing import Tuple

import pandas as pd
import torch
from gym import spaces
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch import Tensor

from bandits.envs import Env


class MovielensEnv(Env):
    def __init__(self, n_arms=10):
        super().__init__(n_arms)
        user_ratings = pd.read_csv("~/Documents/datasets/movielens_100k/user_ratings.csv")
        x = user_ratings[["userId", "rating", "age", "gender", "occupation"]]
        y = user_ratings[["movieId"]]
        cat_cols = make_column_selector(dtype_exclude=["number"])(x)
        num_cols = make_column_selector(dtype_include=["number"])(x)
        x_ppr = ColumnTransformer([
            ("numerical", StandardScaler(), num_cols),
            # ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])
        y_ppr = OneHotEncoder()
        x = x_ppr.fit_transform(x)
        y = y_ppr.fit_transform(y).toarray()
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.action_space = spaces.Discrete(self.y.shape[0])
        self.observation_space = spaces.Box(
            low=self.x.min().item(),
            high=self.x.max().item(),
            shape=(self.x.shape[1], ),
            dtype=np.float32
        )
        self.cursor = 0
        self.rand_idxs = torch.tensor(range(len(self.x)), dtype=torch.long)

    def reset(self) -> Tuple[Tensor, bool]:
        self.cursor = 0
        self.rand_idxs = torch.tensor(range(len(self.x)), dtype=torch.long).multinomial(len(self.x))
        obs, _ = self._fetch_obs()
        return obs, False

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, bool, dict]:
        prev_obs, prev_target = self._fetch_obs()
        rewards = prev_target[actions]
        self.cursor += 1
        next_obs, next_target = self._fetch_obs()
        terminal = self.cursor >= (len(self.x) - 1)
        info = {}
        return next_obs, rewards, terminal, info

    def _fetch_obs(self) -> Tuple[Tensor, Tensor]:
        x = self.x[self.rand_idxs[self.cursor]]
        y = self.y[self.rand_idxs[self.cursor]]
        return x, y
