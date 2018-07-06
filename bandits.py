from abc import ABC, abstractmethod
import numpy as np

NON_STATIONARY_DELTA = 0.001  # how much to shift bandit probabilities


class Bandit(ABC):
    """A single pokie with a fixed payout probability."""

    def __init__(self, is_nonstationary=False) -> None:
        super().__init__()
        self.p: float = None
        self.is_nonstationary: bool = is_nonstationary

    @abstractmethod
    def pull(self) -> float:
        """Take a punt! Is this your lucky day?"""
        pass

    def __repr__(self):
        return 'Bandit<{:0.3f}>'.format(self.p)

    def __lt__(self, other):
        return self.p < other.p


class BernoulliBandit(Bandit):
    """A bandit that will pay out according to a Bernoulli distribution."""

    def __init__(self, is_nonstationary=True) -> None:
        super().__init__(is_nonstationary)
        self.p = np.random.uniform()

    def pull(self) -> float:
        if self.is_nonstationary:  # randomly shift the payout probabilities each pull
            delta = np.random.uniform(low=0 - NON_STATIONARY_DELTA, high=0 + NON_STATIONARY_DELTA)
            # probabilities must stay within [0,1]
            if 0 < self.p + delta < 1:
                self.p += delta

        spin = np.random.uniform()
        reward = 1.0 if spin > self.p else 0.0
        return reward


class GaussianBandit(Bandit):
    """A Gaussian bandit."""

    def __init__(self, is_nonstationary=True) -> None:
        super().__init__(is_nonstationary)
        self.p = np.random.uniform(low=-1.0, high=1.0)  # (uniformly) random gaussian mean [-1, 1]

    def pull(self) -> float:
        # using our preset random mean & unit variance, randomly select from a gaussian
        spin = np.random.normal(self.p, 1.0)
        reward = spin - self.p
        return reward
