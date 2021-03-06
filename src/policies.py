from abc import ABC, abstractmethod
import math
from math import log
import random

import numpy as np


class Policy(ABC):
    """Base class for a policy or method of selecting a bandit."""

    def __init__(self, nb_bandits: int, debug: bool) -> None:
        super().__init__()
        self.nb_bandits: int = nb_bandits
        self.debug = debug

    @abstractmethod
    def select_action(self) -> int:
        """
        Choose a bandit to play.

        :return: the bandit arm index
        """
        pass

    @abstractmethod
    def update(self, bandit: int, reward: float) -> None:
        """
        Update the bandit with reward received.

        :param bandit: the last action taken
        :param reward: the reward received
        """
        pass


class RandomPolicy(Policy):
    """Randomly select an action."""

    def select_action(self) -> int:
        return np.random.choice(self.nb_bandits)

    def update(self, bandit: int, reward: float) -> None:
        """No update on a random policy."""
        pass


class ActionValuePolicy(Policy):
    """Action-Value methods/policies estimate the values of actions and use these estimates to make decisions."""

    def __init__(self, nb_bandits: int, debug: bool, initial_q_value: float = 0.0):
        super().__init__(nb_bandits, debug)
        # number of times we have selected/visited each bandit arm
        self.visits: np.array = np.zeros([nb_bandits], dtype='int32')
        # estimate of the value of each bandit arm
        self.q: np.array = np.array([initial_q_value for _ in range(nb_bandits)])

        if self.debug:
            self.total_rewards = np.zeros([self.nb_bandits])
            self.avg_rewards = np.zeros([self.nb_bandits])
            self.weighted_q = np.zeros([self.nb_bandits])


class EpsilonGreedyPolicy(ActionValuePolicy):
    """Select an action greedily with epsilon chance of a random action."""

    def __init__(self, nb_bandits: int, debug: bool, initial_q_value=0.0, epsilon=0.1):
        super().__init__(nb_bandits, debug, initial_q_value)
        self.epsilon: float = epsilon

    def select_action(self) -> int:
        if np.random.uniform() > self.epsilon:
            bandit_idx = np.argmax(self.q)
        else:
            bandit_idx = np.random.randint(0, self.nb_bandits - 1)
        return int(bandit_idx)

    def update(self, bandit: int, reward: float) -> None:
        self.visits[bandit] += 1
        q_value = self.q[bandit] + (1 / self.visits[bandit]) * (reward - self.q[bandit])
        self.q[bandit] = q_value


class EpsilonGreedyWeightedPolicy(EpsilonGreedyPolicy):
    def __init__(self, nb_bandits: int, debug: bool, initial_q_value=0.0, epsilon=0.1, alpha=0.1):
        super().__init__(nb_bandits, debug, initial_q_value, epsilon)
        self.alpha = alpha

    def update(self, bandit: int, reward: float) -> None:
        self.visits[bandit] += 1
        q_value = self.q[bandit] + self.alpha * (reward - self.q[bandit])
        self.q[bandit] = q_value


class SoftmaxPolicy(ActionValuePolicy):
    def __init__(self, nb_bandits: int, debug: bool, temperature: float = 0.4):
        super().__init__(nb_bandits, debug, initial_q_value=0.0)
        # temperature [0,1] hyperparameter controls how randomly softmax acts
        # higher temperatures (tau -> 1) and all actions tend towards the same probability
        # lower temperatures (tau -> 0) the more likely the action with the highest value will be chosen
        self.tau: float = temperature

    def select_action(self) -> int:
        softmax = self.softmax(self.q)
        bandit_idx = np.random.choice(self.nb_bandits, None, p=softmax)
        return bandit_idx

    def softmax(self, x):
        return np.exp(x / self.tau) / np.sum(np.exp(x / self.tau))

    def update(self, bandit: int, reward: float) -> None:
        self.visits[bandit] += 1

        q_value = self.q[bandit] + (1 / self.visits[bandit]) * (reward - self.q[bandit])
        self.q[bandit] = q_value


class UCB1Policy(ActionValuePolicy):
    """Upper Confidence Bounds."""

    def __init__(self, nb_bandits: int, debug: bool, initial_q_value: float = 0.0):
        super().__init__(nb_bandits, debug, initial_q_value)
        self.total_visits = 0
        self.alpha = 0.1

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


class Exp3Policy(ActionValuePolicy):

    def __init__(self, nb_bandits, debug, initial_q_value=0.0, gamma=0.07):
        super().__init__(nb_bandits, debug, initial_q_value)
        self.gamma = float(gamma)
        self.weights = [1.0] * nb_bandits

    def select_action(self) -> int:
        total_weight = sum(self.weights)
        probs = [0.0] * self.nb_bandits
        for arm in range(self.nb_bandits):
            probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
            probs[arm] = probs[arm] + self.gamma * (1.0 / float(self.nb_bandits))
        value = self._categorical_draw(probs)
        return value

    def update(self, bandit, reward) -> None:
        chosen_arm = bandit
        total_weight = sum(self.weights)
        probs = [0.0] * self.nb_bandits
        for arm in range(self.nb_bandits):
            probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
            probs[arm] = probs[arm] + self.gamma * (1.0 / float(self.nb_bandits))

        x = reward / probs[chosen_arm]

        growth_factor = math.exp((self.gamma / self.nb_bandits) * x)
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
