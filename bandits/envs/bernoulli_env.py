from typing import Tuple

import torch
from torch import Tensor

from bandits.envs.env import Env


class BernoulliEnv(Env):
    def __init__(self, n_arms=10):
        super().__init__(n_arms)
        self.probs = torch.rand(n_arms)

    def reset(self) -> Tuple[Tensor, bool]:
        self.probs = torch.rand(self.n_arms)
        obs = torch.zeros(self.n_arms)
        return obs, False

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, bool, dict]:
        next_obs = torch.zeros(self.n_arms)
        rewards = []
        for a in actions:
            p = torch.rand(1)
            rewards.append(1.0 if p < self.probs[a] else 0.0)
        rewards = torch.tensor(rewards)
        terminal = False
        info = {
            "regret": torch.ones_like(rewards) - rewards,
        }
        return next_obs, rewards, terminal, info
