from typing import List

from gym import Space
import torch
from torch import Tensor

from bandits.bandits.bandit import Bandit
from bandits.transition import Transition


class FixedQLearner(Bandit):
    """Assumes the value of each state-action does not change over time."""

    DEFAULT_CONFIG = {
        "initial_q_value": 0.0,  # optional optimistic initialisation
    }

    def __init__(self, action_space: Space, obs_space: Space, **kwargs) -> None:
        super().__init__(action_space, obs_space, **kwargs)
        self.config.update({
            **FixedQLearner.DEFAULT_CONFIG,  # self config
            **self.DEFAULT_CONFIG,  # child config
            **kwargs,  # custom config (if defined)
        })
        self.q = torch.full((self.n_arms,), self.config["initial_q_value"])
        self.visits = torch.ones(self.n_arms)

    def act(self, obs: Tensor) -> Tensor:
        q = self.q.repeat((obs.shape[0], 1))  # tile Q to same dims as obs batch
        actions = self.sampler.sample(q, self.visits)
        return actions

    def update(self, batch: List[Transition]) -> dict:
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
                self.q[a] = self._update(a, r)
        metrics.update(**{
            "q_min": self.q.min().item(),
            "q_mean": self.q.mean().item(),
            "q_max": self.q.max().item(),
        })

        return metrics

    def _update(self, action: int, reward: float) -> float:
        return self.q[action] + (1 / self.visits[action] * (reward - self.q[action]))


class IncrementalQLearner(FixedQLearner):
    """Incrementally updates state-action values.
    Discount the update by $\alpha$ to account for error in the estimate."""

    DEFAULT_CONFIG = {
        "alpha": 0.1,  # learning-rate
    }

    def _update(self, action: int, reward: float) -> float:
        return self.q[action] + self.alpha * (reward - self.q[action])

    @property
    def alpha(self) -> float:
        return float(self.config["alpha"])
