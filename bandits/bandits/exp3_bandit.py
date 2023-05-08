import math
import random

from bandits.bandits import Bandit


class Exp3Bandit(Bandit):
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

