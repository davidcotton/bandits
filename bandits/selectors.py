from abc import ABC, abstractmethod
import math
from math import log
import random

import numpy as np
import torch
from torch import Tensor


class Selector(ABC):
    DEFAULT_CONFIG = {
        "n_arms": 2,
        "k": 1,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = {  # merge configs
            **Selector.DEFAULT_CONFIG,  # parent config
            **self.DEFAULT_CONFIG,  # child config
            **kwargs,  # custom config (if defined)
        }
        self.n_arms = int(self.config["n_arms"])
        self.k = int(self.config["k"])

    @abstractmethod
    def select(self, probs: Tensor) -> Tensor:
        pass

    @property
    def n_arms(self) -> int:
        return int(self.config["n_arms"])

    @property
    def k(self) -> int:
        return int(self.config["k"])


class RandomSelector(Selector):
    """Randomly select an action."""
    def select(self, probs: Tensor) -> Tensor:
        return np.random.choice(self.n_arms)


class ActionValueSelector(Selector):
    """Action-Value methods/policies estimate the values of actions and use these estimates to make decisions."""
    DEFAULT_CONFIG = {
        "initial_q_value": 0.0,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # number of times we have selected/visited each bandit arm
        self.visits: np.array = np.zeros([self.n_arms], dtype=np.int)
        # estimate of the value of each bandit arm
        self.q: np.array = np.array([self.initial_q_value for _ in range(self.n_arms)])

        if self.debug:
            self.total_rewards = np.zeros([self.nb_bandits])
            self.avg_rewards = np.zeros([self.nb_bandits])
            self.weighted_q = np.zeros([self.nb_bandits])

    @property
    def initial_q_value(self) -> float:
        return float(self.config["initial_q_value"])


class EpsilonGreedySelector(ActionValueSelector):
    """Select an action greedily with epsilon chance of a random action."""
    DEFAULT_CONFIG = {
        "epsilon": 0.1,
    }

    def select_action(self) -> int:
        if np.random.uniform() > self.epsilon:
            bandit_idx = np.argmax(self.q)
        else:
            bandit_idx = np.random.randint(0, self.n_arms - 1)
        return int(bandit_idx)

    def update(self, bandit: int, reward: float) -> None:
        self.visits[bandit] += 1
        q_value = self.q[bandit] + (1 / self.visits[bandit]) * (reward - self.q[bandit])
        self.q[bandit] = q_value

    @property
    def epsilon(self) -> float:
        return float(self.config["epsilon"])


class EpsilonGreedyWeightedSelector(EpsilonGreedySelector):
    DEFAULT_CONFIG = {
        "epsilon": 0.1,
        "alpha": 0.1,
    }

    def update(self, bandit: int, reward: float) -> None:
        self.visits[bandit] += 1
        q_value = self.q[bandit] + self.alpha * (reward - self.q[bandit])
        self.q[bandit] = q_value


class SoftmaxSelector(ActionValueSelector):
    DEFAULT_CONFIG = {
        # temperature [0,1] hyperparameter controls how randomly softmax acts
        # higher temperatures (tau -> 1) and all actions tend towards the same probability
        # lower temperatures (tau -> 0) the more likely the action with the highest value will be chosen
        "tau": 0.4,
    }

    def select_action(self) -> int:
        softmax = self.softmax(self.q)
        bandit_idx = np.random.choice(self.n_arms, None, p=softmax)
        return bandit_idx

    def softmax(self, x):
        return np.exp(x / self.tau) / np.sum(np.exp(x / self.tau))

    def update(self, bandit: int, reward: float) -> None:
        self.visits[bandit] += 1
        q_value = self.q[bandit] + (1 / self.visits[bandit]) * (reward - self.q[bandit])
        self.q[bandit] = q_value

    @property
    def tau(self) -> float:
        return float(self.config["tau"])


class UCB1Selector(ActionValueSelector):
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


class Exp3Selector(ActionValueSelector):
    DEFAULT_CONFIG = {
        "gamma": 0.07,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights = [1.0] * self.n_arms

    def select_action(self) -> int:
        total_weight = sum(self.weights)
        probs = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
            probs[arm] = probs[arm] + self.gamma * (1.0 / float(self.n_arms))
        value = self._categorical_draw(probs)
        return value

    def update(self, bandit, reward) -> None:
        chosen_arm = bandit
        total_weight = sum(self.weights)
        probs = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
            probs[arm] = probs[arm] + self.gamma * (1.0 / float(self.n_arms))

        x = reward / probs[chosen_arm]

        growth_factor = math.exp((self.gamma / self.n_arms) * x)
        self.weights[chosen_arm] = self.weights[chosen_arm] * growth_factor

    @staticmethod
    def _categorical_draw(probs):
        z = random.random()
        cumulative_probs = 0.0
        for i in range(len(probs)):
            cumulative_probs += probs[i]
            if cumulative_probs > z:
                return i
        return len(probs) - 1

    @property
    def gamma(self) -> float:
        return float(self.config["gamma"])
