from abc import ABC, abstractmethod
from typing import List, Mapping

from torch import Tensor

from bandits.samplers import (
    EpsilonGreedySampler,
    Sampler,
)
from bandits.transition import Transition


class Bandit(ABC):
    DEFAULT_CONFIG = {
        "n_arms": 2,
        # "k": 1,
        "sampler": {
            # "class": GreedySampler,
            "class": EpsilonGreedySampler,
        },
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.config = {  # merge configs
            **Bandit.DEFAULT_CONFIG,  # parent config
            **self.DEFAULT_CONFIG,  # child config
            **kwargs,  # custom config (if defined)
        }
        sampler_config = self.config["sampler"]
        self.sampler: Sampler = sampler_config["class"](**sampler_config.get("params", {}))

    @abstractmethod
    def act(self, obs: Tensor) -> Tensor:
        pass

    def update(self, batch: List[Transition]) -> dict:
        return {}

    @property
    def n_arms(self) -> int:
        return int(self.config["n_arms"])


Bandits = Mapping[str, Bandit]
