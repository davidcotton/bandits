import importlib
from abc import ABC, abstractmethod
from typing import List, Mapping

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
        "n_arms": 2,
        # "k": 1,
        "sampler": {
            # "class": GreedySampler,
            "class": EpsilonGreedySampler,
            "params": {"epsilon": 0.1},
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
        return int(self.config["n_arms"])


Bandits = Mapping[str, Bandit]
