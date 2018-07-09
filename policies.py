from abc import ABC, abstractmethod
from math import log
import numpy as np


class Policy(ABC):
    """Base class for a policy or method of selecting a bandit."""

    def __init__(self, nb_bandits: int) -> None:
        super().__init__()
        self.nb_bandits: int = nb_bandits

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

    def __init__(self, nb_bandits: int, initial_q_value: float = 0.0):
        super().__init__(nb_bandits)
        # number of times we have selected/visited each bandit arm
        self.visits: np.array = np.zeros([nb_bandits], dtype='int32')
        # estimate of the value of each bandit arm
        self.q: np.array = np.array([initial_q_value for _ in range(nb_bandits)])


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
        return int(bandit_idx)

    def update(self, bandit: int, reward: float) -> None:
        self.visits[bandit] += 1
        q_value = self.q[bandit] + (1 / self.visits[bandit]) * (reward - self.q[bandit])
        self.q[bandit] = q_value


class EpsilonGreedyWeightedPolicy(EpsilonGreedyPolicy):
    def __init__(self, bandits=None, initial_q_value=0.0, epsilon=0.1, alpha=0.1):
        super().__init__(bandits, initial_q_value, epsilon)
        self.alpha = alpha

    def update(self, bandit: int, reward: float) -> None:
        self.visits[bandit] += 1
        q_value = self.q[bandit] + self.alpha * (reward - self.q[bandit])
        self.q[bandit] = q_value


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
        return bandit_idx

    def softmax(self, x):
        return np.exp(x / self.tau) / np.sum(np.exp(x / self.tau))

    def update(self, bandit: int, reward: float) -> None:
        self.visits[bandit] += 1

        q_value = self.q[bandit] + (1 / self.visits[bandit]) * (reward - self.q[bandit])
        self.q[bandit] = q_value


class UCB1Policy(ActionValuePolicy):
    """Upper Confidence Bounds."""

    def __init__(self, nb_bandits: int, initial_q_value: float = 0.0):
        super().__init__(nb_bandits, initial_q_value)
        self.total_visits = 0

    def select_action(self) -> int:
        self.total_visits += 1

        # initialise: make sure we have visited every action
        if np.min(self.visits) == 0:
            for idx, count in enumerate(self.visits):
                if count == 0:
                    return idx

        ucb_values = self.q / self.visits + self.upper_bound(self.total_visits, self.visits)

        return int(np.argmax(ucb_values))

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

        q_value = self.q[bandit] + reward
        # q_value = self.q[bandit] + (1 / self.visits[bandit]) * (reward - self.q[bandit])
        self.q[bandit] = q_value
