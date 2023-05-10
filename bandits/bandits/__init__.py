__all__ = [
    "Bandit",
    "Bandits",
    "Sampler",
    "EpsilonGreedySampler",
    "GreedySampler",
    "IncrementalQLearner",
    "RandomSampler",
    "SoftmaxSampler",
    "UCBSampler",
]

import importlib

from bandits.bandits.bandit import (
    Bandit,
    Bandits,
)
from bandits.bandits.q_bandit import (
    FixedQLearner,
    IncrementalQLearner,
)
from bandits.bandits.samplers import (
    Sampler,
    EpsilonGreedySampler,
    GreedySampler,
    RandomSampler,
    SoftmaxSampler,
    UCBSampler,
)
from bandits.utils import extract_module


def get_bandits(config):
    model_configs = config["models"]
    bandits = {}
    for name, cfg in model_configs.items():
        namespace, import_ = extract_module(cfg["class"], "bandits.bandits")
        model_cls = getattr(importlib.import_module(namespace), import_)
        model_params = cfg.get("params", {})
        # bandits[name] = model_cls(**model_params)
        bandits[name] = model_cls, model_params
    return bandits
