from abc import ABC, abstractmethod
from typing import List

import torch
from torch import Tensor

from bandits.bandits.bandit import Bandit
from bandits.samplers import EpsilonGreedySampler
from bandits.transition import Transition


class AbstractQLearner(Bandit, ABC):
    """Q-learner abstract base class."""

    DEFAULT_CONFIG = {
        "initial_q_value": 0.0,  # optional optimistic initialisation
        "sampler": {
            "class": EpsilonGreedySampler,
            "params": {"epsilon": 0.1},
        },
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.q = torch.full((self.n_arms,), self.config["initial_q_value"])

    def act(self, obs: Tensor) -> Tensor:
        q = self.q.repeat((obs.shape[0], 1))  # tile Q to same dims as obs batch
        actions = self.sampler.sample(q)
        return actions

    def update(self, batch: List[Transition]) -> dict:
        metrics = super().update(batch)
        metrics.update(self._update(batch))
        metrics.update(**{
            "q_min": self.q.min(),
            "q_mean": self.q.mean(),
            "q_max": self.q.max(),
        })
        return metrics

    @abstractmethod
    def _update(self, batch: List[Transition]) -> dict:
        return {}


class FixedQLearner(AbstractQLearner):
    """Assumes the value of each state-action does not change over time."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.visits = torch.ones(self.n_arms)

    def _update(self, batch: List[Transition]) -> dict:
        metrics = super().update(batch)
        for _, actions, _, rewards, _ in batch:
            self.visits.scatter_add_(
                dim=0, index=actions, src=torch.ones_like(actions, dtype=self.visits.dtype)
            )
            if actions.ndim == 0:
                actions = actions.unsqueeze(0)
            if rewards.ndim == 0:
                rewards = rewards.unsqueeze(0)
            for a, r in zip(actions, rewards):
                q = self.q[a] + (1 / self.visits[a] * (r - self.q[a]))
                self.q[a] = q
        return metrics


class IncrementalQLearner(AbstractQLearner):
    """Incrementally updates state-action values.
    Discount the update by $\alpha$ to account for error in the estimate."""

    DEFAULT_CONFIG = {
        "alpha": 0.1,  # learning-rate
    }

    def _update(self, batch: List[Transition]) -> dict:
        metrics = super().update(batch)
        for _, actions, _, rewards, _ in batch:
            for a, r in zip(actions, rewards):
                self.q[a] = self.q[a] + self.alpha * (r - self.q[a])
        return metrics

    @property
    def alpha(self) -> float:
        return float(self.config["alpha"])
