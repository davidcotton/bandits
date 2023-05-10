from typing import Tuple

import torch
from torch import Tensor

from bandits.envs.env import Env


class GaussianEnv(Env):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.probs = torch.rand(n_arms)

    def reset(self) -> Tuple[Tensor, bool]:
        pass

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, bool, dict]:
        pass
