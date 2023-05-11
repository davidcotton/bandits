__all__ = [
    "Env",
    "BernoulliEnv",
    "GaussianEnv",
    "MovielensEnv",
    "build_env",
]

import importlib

from bandits.envs.bernoulli_env import BernoulliEnv
from bandits.envs.env import Env
from bandits.envs.gaussian_env import GaussianEnv
from bandits.envs.movielens_env import MovielensEnv
from bandits.utils import extract_module


def build_env(config) -> Env:
    env_config = config["environment"]
    namespace, import_ = extract_module(env_config["class"], "bandits.envs")
    env_cls = getattr(importlib.import_module(namespace), import_)
    env_params = {
        **env_config.get("params", {}),
    }
    return env_cls(env_params)
