from typing import List

import torch
from torch import Tensor

from bandits.bandits.bandit import Bandit
from bandits.samplers import EpsilonGreedySampler
from bandits.transition import Transition


class QBandit(Bandit):
    DEFAULT_CONFIG = {
        "initial_q_value": 0.0,
        "sampler": {
            "class": EpsilonGreedySampler,
            "params": {"epsilon": 0.1},
        },
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q = torch.full((self.n_arms,), self.config["initial_q_value"])
        self.visits = torch.ones(self.n_arms)

    def act(self, obs: Tensor) -> Tensor:
        q = self.q.repeat((obs.shape[0], 1))  # tile Q to same dims as obs batch
        actions = self.sampler.sample(q)
        return actions

    def update(self, batch: List[Transition]) -> dict:
        metrics = super().update(batch)
        for _, actions, _, rewards, _ in batch:
            self.visits.scatter_add_(
                dim=0, index=actions, src=torch.ones_like(actions, dtype=self.visits.dtype)
            )
            for a, r in zip(actions, rewards):
                q = self.q[a] + (1 / self.visits[a] * (r - self.q[a]))
                self.q[a] = q
        return metrics


class WeightedQBandit(QBandit):
    DEFAULT_CONFIG = {
        "alpha": 0.1,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q = torch.full((self.n_arms,), self.config["initial_q_value"])

    def update(self, batch: List[Transition]) -> dict:
        metrics = super().update(batch)
        for _, actions, _, rewards, _ in batch:
            for a, r in zip(actions, rewards):
                self.q[a] = self.q[a] + self.alpha * (r - self.q[a])
        return metrics

    @property
    def alpha(self) -> float:
        return float(self.config["alpha"])
