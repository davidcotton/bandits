from abc import ABC, abstractmethod
from typing import List, Mapping

import torch
from torch import Tensor

from bandits.transition import Transition


class Bandit(ABC):
    DEFAULT_CONFIG = {
        "n_arms": 2,
        # "k": 1,
        # "actor": GreedySelector,
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.config = {  # merge configs
            **Bandit.DEFAULT_CONFIG,  # parent config
            **self.DEFAULT_CONFIG,  # child config
            **kwargs,  # custom config (if defined)
        }

    @abstractmethod
    def act(self, obs: Tensor) -> Tensor:
        pass

    def update(self, batch: List[Transition]) -> dict:
        return {}

    @property
    def n_arms(self) -> int:
        return int(self.config["n_arms"])


Bandits = Mapping[str, Bandit]


class SimpleQBandit(Bandit):
    DEFAULT_CONFIG = {
        # "selector": EpsilonGreedySelector,
        "epsilon": 0.1,
        "alpha": 0.1,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q = torch.zeros(self.n_arms)
        self.arm_visits = torch.ones(self.n_arms)

    def act(self, obs: Tensor) -> Tensor:
        pass

    def update(self, batch: List[Transition]) -> dict:
        return super().update(batch)
