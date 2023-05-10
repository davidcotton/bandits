from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor


class Sampler(ABC):
    DEFAULT_CONFIG = {
        "n_arms": 2,
        "k": 1,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = {  # merge configs
            **Sampler.DEFAULT_CONFIG,  # parent config
            **self.DEFAULT_CONFIG,  # child config
            **kwargs,  # custom config (if defined)
        }
        # self.n_arms = int(self.config["n_arms"])
        # self.k = int(self.config["k"])

    @abstractmethod
    def sample(self, logits: Tensor, visits: Tensor) -> Tensor:
        pass

    @property
    def n_arms(self) -> int:
        return int(self.config["n_arms"])

    @property
    def k(self) -> int:
        return int(self.config["k"])


class RandomSampler(Sampler):
    """Randomly sample an action."""
    def sample(self, logits: Tensor, visits: Tensor) -> Tensor:
        return np.random.choice(self.n_arms)


class GreedySampler(Sampler):
    def sample(self, logits: Tensor, visits: Tensor) -> Tensor:
        _, actions = logits.topk(k=self.k)
        return actions


class EpsilonGreedySampler(Sampler):
    """Select an action greedily with epsilon chance of a random action."""
    DEFAULT_CONFIG = {
        "epsilon": 0.1,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rand_actions_dist = torch.ones(self.n_arms)

    def sample(self, logits: Tensor, visits: Tensor) -> Tensor:
        batch_size = logits.shape[0]
        # boolean vector mask of which actions are greedy (1) vs. random (0)
        random_masks = torch.gt(torch.rand(size=(batch_size, self.k)), self.epsilon).int()
        # vector of random actions
        rand_actions_dist = self.rand_actions_dist.repeat((batch_size, 1))  # expand to batch size
        # sample w/o replacement
        random_actions = rand_actions_dist.multinomial(self.k, replacement=False)
        # vector of greedy actions
        _, greedy_actions = logits.topk(k=self.k)
        # merge random & greedy actions
        actions = (greedy_actions * random_masks) + ((random_masks ^ 1) * random_actions)
        return actions

    @property
    def epsilon(self) -> float:
        return float(self.config["epsilon"])


class SoftmaxSampler(Sampler):
    DEFAULT_CONFIG = {
        # temperature [0,1] hyperparameter controls how randomly softmax acts
        # higher temperatures (tau -> 1) and all actions tend towards the same probability
        # lower temperatures (tau -> 0) the more likely the action with the highest value will be chosen
        "tau": 0.4,
    }

    def sample(self, logits: Tensor, visits: Tensor) -> Tensor:
        return torch.softmax(logits / self.tau, dim=1)

    @property
    def tau(self) -> float:
        return float(self.config["tau"])


class UCBSampler(Sampler):
    DEFAULT_CONFIG = {
        "confidence": 2.0,
    }

    def sample(self, logits: Tensor, visits: Tensor) -> Tensor:
        log_total_visits = self.confidence * torch.log(visits.sum())
        actions_batch = []
        for batch in logits:
            uncertainty = torch.sqrt(log_total_visits / visits)
            ucb = batch + uncertainty
            _, actions = ucb.topk(k=self.k)
            actions_batch.append(actions)
        return torch.stack(actions_batch)

    @property
    def confidence(self) -> float:
        return float(self.config["confidence"])
