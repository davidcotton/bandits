import numpy as np
from typing import Tuple

import pandas as pd
import scipy.sparse
import torch
from gym import spaces
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch import Tensor

from bandits.envs import Env


class MovielensEnv(Env):
    def __init__(self, n_arms=10):
        super().__init__(n_arms)
        users = pd.read_csv(
            "~/Documents/data/ml-100k/u.user",
            sep="|",
            header=None,
            names=["user_id", "age", "gender", "occupation", "zip_code"],
        )
        ratings = pd.read_csv(
            "~/Documents/data/ml-100k/u.data",
            sep="\t",
            header=None,
            names=["user_id", "movie_id", "rating", "unix_timestamp"],
            converters={
                "unix_timestamp": lambda ts: pd.Timestamp(int(ts), unit="s").to_datetime64(),
            }
        )
        user_ratings = ratings.merge(users, how="left", on="user_id")
        # user_ratings = pd.read_csv("~/Documents/data/movielens-small/ratings.csv")
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

    def reset(self) -> Tuple[Tensor, bool]:
        self.cursor = 0
        # self.rand_idxs = torch.randperm(n=len(self.x))
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
