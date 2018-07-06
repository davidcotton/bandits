from abc import ABC, abstractmethod
import numpy as np
from typing import List


class Policy(ABC):
    """Base class for a policy or method of selecting a bandit."""

    def __init__(self, nb_bandits: int) -> None:
        super().__init__()
        self.nb_bandits: int = nb_bandits

    @abstractmethod
    def select_action(self) -> int:
        """Choose a bandit and pull it's level.
        Returns the reward."""
        pass

    @abstractmethod
    def update(self, reward: float) -> None:
        """Update the bandit with reward received."""
        pass


class RandomPolicy(Policy):
    """Randomly select an action."""

    def select_action(self) -> int:
        return np.random.choice(self.nb_bandits)

    def update(self, reward: float) -> None:
        """No update on a random policy."""
        pass


class ActionValuePolicy(Policy):
    """Action-Value methods/policies estimate the values of actions and use these estimates to make decisions."""

    def __init__(self, nb_bandits: int, initial_q_value: float = 0.0):
        super().__init__(nb_bandits)
        # the estimates of values of each bandit
        self.q: np.array = np.array([initial_q_value for _ in range(nb_bandits)])
        # the number of times we have selected each bandit
        # self.counts: List[int] = [0 for _ in range(nb_bandits)]
        self.counts: np.array = np.zeros([nb_bandits], dtype='int32')
        self.prev_action = None


class EpsilonGreedyPolicy(ActionValuePolicy):
    """Select an action greedily with epsilon chance of a random action."""

    def __init__(self, nb_bandits: int, initial_q_value=0.0, epsilon=0.1):
        super().__init__(nb_bandits, initial_q_value)
        self.epsilon: float = epsilon

    def select_action(self) -> int:
        if np.random.uniform() > self.epsilon:
            bandit_idx = np.argmax(self.q)
        else:
            bandit_idx = np.random.randint(0, self.nb_bandits - 1)
        self.prev_action = bandit_idx
        return int(bandit_idx)

    def update(self, reward: float) -> None:
        idx = self.prev_action
        self.counts[idx] += 1
        q_value = self.q[idx] + (1 / self.counts[idx]) * (reward - self.q[idx])
        self.q[idx] = q_value


class EpsilonGreedyWeightedPolicy(EpsilonGreedyPolicy):
    def __init__(self, bandits=None, initial_q_value=0.0, epsilon=0.1, alpha=0.1):
        super().__init__(bandits, initial_q_value, epsilon)
        self.alpha = alpha

    def update(self, reward: float) -> None:
        idx = self.prev_action
        self.counts[idx] += 1
        q_value = self.q[idx] + self.alpha * (reward - self.q[idx])
        self.q[idx] = q_value


class SoftmaxPolicy(ActionValuePolicy):
    def __init__(self, nb_bandits: int, temperature: float = 0.4):
        super().__init__(nb_bandits, initial_q_value=0.0)
        # temperature [0,1] hyperparameter controls how randomly softmax acts
        # higher temperatures (tau -> 1) and all actions tend towards the same probability
        # lower temperatures (tau -> 0) the more likely the action with the highest value will be chosen
        self.tau: float = temperature

    def select_action(self) -> int:
        softmax = self.softmax(self.q)
        bandit_idx = np.random.choice(self.nb_bandits, None, p=softmax)
        self.prev_action = bandit_idx
        return bandit_idx

    def softmax(self, x):
        return np.exp(x / self.tau) / np.sum(np.exp(x / self.tau))

    def update(self, reward: float) -> None:
        idx = self.prev_action
        self.counts[idx] += 1

        q_value = self.q[idx] + (1 / self.counts[idx]) * (reward - self.q[idx])
        self.q[idx] = q_value
