from typing import Tuple

from gym import spaces
import torch
from torch import Tensor

from bandits.envs.env import Env


class BernoulliEnv(Env):
    def __init__(self, env_config=None):
        super().__init__(env_config)
        n_arms = env_config.get("n_arms", 10)
        self.probs = torch.rand(n_arms)
        self.action_space = spaces.Discrete(n_arms)
        # self.observation_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            # dtype=torch.float
        )

    def reset(self) -> Tuple[Tensor, bool]:
        self.probs = torch.rand_like(self.probs)
        return self._fetch_obs(), False

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, bool, dict]:
        next_obs = self._fetch_obs()
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

    def _fetch_obs(self) -> Tensor:
        return torch.zeros(self.observation_space.shape)
