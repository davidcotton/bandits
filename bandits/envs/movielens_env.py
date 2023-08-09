import numpy as np
from typing import Tuple

import scipy.sparse
import torch
from gym import spaces
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch import Tensor

from bandits.datasets import MovieLensDataset
from bandits.envs import Env


class MovieLensEnv(Env):
    def __init__(self, n_arms=10, dataset_name="ml-100k", shuffle_data=False):
        super().__init__(n_arms)
        user_ratings = MovieLensDataset(dataset_name)()
        x = user_ratings[["user_id", "rating", "age", "gender", "occupation"]]
        y = user_ratings[["movie_id"]]
        cat_cols = make_column_selector(dtype_exclude=["number"])(x)
        num_cols = make_column_selector(dtype_include=["number"])(x)
        x_ppr = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            # ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])
        x = x_ppr.fit_transform(x)
        if scipy.sparse.issparse(x):
            x = x.toarray()
        y_ppr = OneHotEncoder()
        y = y_ppr.fit_transform(y).toarray()
        self.n_arms = len(y_ppr.get_feature_names_out())
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.observation_space = spaces.Box(
            low=self.x.min().item(),
            high=self.x.max().item(),
            shape=(self.x.shape[1],),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_arms)
        self.cursor = 0
        self.rand_idxs = torch.tensor(range(len(self.x)), dtype=torch.long)
        self.shuffle_data = bool(shuffle_data)

    def reset(self) -> Tuple[Tensor, bool]:
        self.cursor = 0
        if self.shuffle_data:
            self.rand_idxs = torch.randperm(n=len(self.x))
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
