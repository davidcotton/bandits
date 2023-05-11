import importlib
from abc import ABC, abstractmethod
from typing import List, Mapping

from gym import Space, spaces
from torch import Tensor

from bandits.bandits.samplers import (
    Sampler,
    EpsilonGreedySampler,
    GreedySampler,
)
from bandits.transition import Transition
from bandits.utils import extract_module


class Bandit(ABC):
    DEFAULT_CONFIG = {
        # "k": 1,
        "sampler": {
            # "class": GreedySampler,
            "class": EpsilonGreedySampler,
            "params": {"epsilon": 0.1},
        },
    }

    def __init__(self, action_space: Space, obs_space: Space, **kwargs):
        super().__init__()
        self.config = {  # merge configs
            **Bandit.DEFAULT_CONFIG,  # parent config
            **self.DEFAULT_CONFIG,  # child config
            **kwargs,  # custom config (if defined)
        }
        self.obs_space: Space = obs_space
        self.action_space: Space = action_space
        sampler_config = self.config["sampler"]
        sampler_cls = sampler_config["class"]
        if isinstance(sampler_cls, str):
            namespace, import_ = extract_module(sampler_cls, "bandits.bandits")
            sampler_cls = getattr(importlib.import_module(namespace), import_)
        self.sampler: Sampler = sampler_cls(**sampler_config.get("params", {}))

    @abstractmethod
    def act(self, obs: Tensor) -> Tensor:
        pass

    def update(self, batch: List[Transition]) -> dict:
        return {}

    @property
    def n_arms(self) -> int:
        if isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        else:
            raise ValueError("Unsupported action space")


Bandits = Mapping[str, Bandit]
