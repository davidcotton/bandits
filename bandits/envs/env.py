from abc import abstractmethod
from typing import Tuple

import gym
from torch import Tensor


class Env(gym.Env):
    def __init__(self, env_config=None) -> None:
        super().__init__()
        self.env_config = env_config or {}

    @abstractmethod
    def reset(self) -> Tuple[Tensor, bool]:
        pass

    @abstractmethod
    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, bool, dict]:
        pass


# NON_STATIONARY_DELTA = 0.001  # how much to shift bandit probabilities
#
#
# class Bandit(ABC):
#     """A single pokie with a fixed payout probability."""
#
#     def __init__(self, p: float, is_nonstationary=False) -> None:
#         super().__init__()
#         self.p: float = p
#         self.is_nonstationary: bool = is_nonstationary
#
#     @abstractmethod
#     def pull(self) -> float:
#         """Take a punt! Is this your lucky day?"""
#         pass
#
#     @staticmethod
#     def build(nb_bandits) -> list:
#         """Static factory method to implemented on each subclass."""
#         pass
#
#     def __repr__(self):
#         return f'{self.__class__.__name__}({self.p:0.3f})'
#
#     def __lt__(self, other):
#         return self.p < other.p
#
#
# class BernoulliBandit(Bandit):
#     """A bandit that will pay out according to a Bernoulli distribution."""
#
#     def pull(self) -> float:
#         if self.is_nonstationary:  # randomly shift the payout probabilities each pull
#             delta = np.random.uniform(low=0 - NON_STATIONARY_DELTA, high=0 + NON_STATIONARY_DELTA)
#             # probabilities must stay within [0,1]
#             if 0 < self.p + delta < 1:
#                 self.p += delta
#
#         spin = np.random.uniform()
#         # reward = 1.0 if spin > self.p else 0.0
#         reward = 1.0 if spin < self.p else 0.0
#         return reward
#
#     @staticmethod
#     def build(nb_bandits) -> List[Bandit]:
#         """Build a group of Bernoulli bandits."""
#         logits = np.random.uniform(low=0.0, high=1.0, size=nb_bandits)
#         bandits = [BernoulliBandit(p, is_nonstationary=False) for p in logits]
#         return bandits
#
#
# class GaussianBandit(Bandit):
#     """A Gaussian bandit.
#     A group of gaussian bandits collectively have a zero-mean and unit variance.
#     Each gaussian bandit has its own random mean from which rewards come."""
#
#     def pull(self) -> float:
#         # using our preset random mean & unit variance, randomly sample from a gaussian
#         spin = np.random.normal(self.p, 1.0)
#         return spin
#
#     @staticmethod
#     def build(nb_bandits) -> List[Bandit]:
#         """Build a collection of Gaussian bandits."""
#         mean, stddev = 0.0, 1.0
#         logits = np.random.normal(mean, stddev, nb_bandits)
#         normalised_probs = (logits - logits.mean(axis=0)) / logits.std(axis=0)
#         bandits = [GaussianBandit(p, is_nonstationary=False) for p in normalised_probs]
#         return bandits
