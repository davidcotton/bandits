__all__ = [
    "Bandit",
    "Bandits",
    "Sampler",
    "EpsilonGreedySampler",
    "GreedySampler",
    "IncrementalQLearner",
    "LinUCB",
    "RandomSampler",
    "SoftmaxSampler",
    "UCBSampler",
    "build_bandits",
]

import importlib

from bandits.bandits.bandit import (
    Bandit,
    Bandits,
)
from bandits.bandits.lin_ucb import LinUCB
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
from bandits.envs import Env
from bandits.utils import extract_module


def build_bandits(config: dict, env: Env):
    model_configs = config["models"]
    bandits = {}
    for name, cfg in model_configs.items():
        namespace, import_ = extract_module(cfg["class"], "bandits.bandits")
        model_cls = getattr(importlib.import_module(namespace), import_)
        model_params = {
            **cfg.get("params", {}),
        }
        bandits[name] = model_cls(env.action_space, env.observation_space, **model_params)
        # bandits[name] = model_cls, model_params
    return bandits
