from abc import ABC, abstractmethod
from math import log

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
    def sample(self, probs: Tensor) -> Tensor:
        pass

    @property
    def n_arms(self) -> int:
        return int(self.config["n_arms"])

    @property
    def k(self) -> int:
        return int(self.config["k"])


class RandomSampler(Sampler):
    """Randomly sample an action."""
    def sample(self, probs: Tensor) -> Tensor:
        return np.random.choice(self.n_arms)


# class ActionValueSampler(Sampler):
#     """Action-Value methods/policies estimate the values of actions and use these estimates to make decisions."""
#     DEFAULT_CONFIG = {
#         "initial_q_value": 0.0,
#     }
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         # number of times we have selected/visited each bandit arm
#         self.visits: np.array = np.zeros([self.n_arms], dtype=np.int)
#         # estimate of the value of each bandit arm
#         self.q: np.array = np.array([self.initial_q_value for _ in range(self.n_arms)])
#
#         if self.debug:
#             self.total_rewards = np.zeros([self.nb_bandits])
#             self.avg_rewards = np.zeros([self.nb_bandits])
#             self.weighted_q = np.zeros([self.nb_bandits])
#
#     @property
#     def initial_q_value(self) -> float:
#         return float(self.config["initial_q_value"])


class GreedySampler(Sampler):
    def sample(self, probs: Tensor) -> Tensor:
        _, actions = probs.topk(k=1)
        return actions


class EpsilonGreedySampler(Sampler):
    """Select an action greedily with epsilon chance of a random action."""
    DEFAULT_CONFIG = {
        "epsilon": 0.1,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rand_actions_dist = torch.ones(self.n_arms)

    def sample(self, probs: Tensor) -> Tensor:
        batch_size = probs.shape[0]
        # boolean vector mask of which actions are greedy (1) vs. random (0)
        random_masks = torch.gt(torch.rand(size=(batch_size, self.k)), self.epsilon).int()
        # vector of random actions
        rand_actions_dist = self.rand_actions_dist.repeat((batch_size, 1))  # expand to batch size
        # sample w/o replacement
        random_actions = rand_actions_dist.multinomial(self.k, replacement=False)
        # vector of greedy actions
        _, greedy_actions = probs.topk(k=self.k)
        # merge random & greedy actions
        actions = (greedy_actions * random_masks) + ((random_masks ^ 1) * random_actions)
        return actions

    @property
    def epsilon(self) -> float:
        return float(self.config["epsilon"])

    # def select_action(self) -> int:
    #     if np.random.uniform() > self.epsilon:
    #         bandit_idx = np.argmax(self.q)
    #     else:
    #         bandit_idx = np.random.randint(0, self.n_arms - 1)
    #     return int(bandit_idx)
    #
    # def update(self, bandit: int, reward: float) -> None:
    #     self.visits[bandit] += 1
    #     q_value = self.q[bandit] + (1 / self.visits[bandit]) * (reward - self.q[bandit])
    #     self.q[bandit] = q_value


# class EpsilonGreedyWeightedSampler(EpsilonGreedySampler):
#     DEFAULT_CONFIG = {
#         "epsilon": 0.1,
#         "alpha": 0.1,
#     }
#
#     def update(self, bandit: int, reward: float) -> None:
#         self.visits[bandit] += 1
#         q_value = self.q[bandit] + self.alpha * (reward - self.q[bandit])
#         self.q[bandit] = q_value


class SoftmaxSampler(Sampler):
    DEFAULT_CONFIG = {
        # temperature [0,1] hyperparameter controls how randomly softmax acts
        # higher temperatures (tau -> 1) and all actions tend towards the same probability
        # lower temperatures (tau -> 0) the more likely the action with the highest value will be chosen
        "tau": 0.4,
    }

    def sample(self, probs: Tensor) -> Tensor:
        return torch.softmax(probs / self.tau, dim=1)

    @property
    def tau(self) -> float:
        return float(self.config["tau"])


class UCB1Sampler(Sampler):
    """Upper Confidence Bounds."""
    DEFAULT_CONFIG = {
        "alpha": 0.1,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_visits = 0

    def select_action(self) -> int:
        self.total_visits += 1

        # initialise: make sure we have visited every action
        if np.min(self.visits) == 0:
            for idx, count in enumerate(self.visits):
                if count == 0:
                    break
        else:
            ucb_values = self.q / self.visits + self.upper_bound(self.total_visits, self.visits)
            idx = int(np.argmax(ucb_values))

        return idx

    @staticmethod
    def upper_bound(step: int, visits: np.array) -> float:
        """Calculate the upper confidence bound for a vector of bandit arms.

        :param step: the total number of visits to ALL arms
        :param visits: a vector of number of visits to each arm
        :return: the UCB value
        """
        return np.sqrt(2 * log(step) / visits)

    def update(self, bandit: int, reward: float) -> None:
        self.visits[bandit] += 1
        self.total_rewards[bandit] += reward

        total_reward = self.q[bandit] + reward
        # avg_reward = self.total_rewards[bandit] / self.visits[bandit]
        avg_reward = self.q[bandit] + (1 / self.visits[bandit]) * (reward - self.q[bandit])
        weighted_q = self.q[bandit] + self.alpha * (reward - self.q[bandit])

        if self.debug:
            self.total_rewards[bandit] = total_reward
            self.avg_rewards[bandit] = avg_reward
            self.weighted_q[bandit] = weighted_q

        self.q[bandit] = weighted_q

    @property
    def alpha(self) -> float:
        return float(self.config["alpha"])
