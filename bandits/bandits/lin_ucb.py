from typing import List

import torch
from gym import Space
from torch import Tensor

from bandits.bandits.bandit import Bandit
from bandits.transition import Transition


class RidgeRegressor:
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=None, tol=1e-4,) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = max_iter
        self.tol = float(tol)
        self.weights = None

    def fit(self, x: Tensor, y: Tensor) -> None:
        assert x.shape[0] == y.shape[0]
        if self.fit_intercept:
            x = torch.cat([torch.ones(x.shape[0], 1), x], dim=1)
        lhs = x.T @ x
        rhs = x.T @ y
        if self.alpha == 0.0:
            self.weights, _ = torch.linalg.lstsq(rhs, lhs)
        else:
            ridge = self.alpha * torch.eye(lhs.shape[0])
            self.weights, _ = torch.linalg.lstsq(rhs, lhs + ridge)

    def predict(self, x: Tensor) -> Tensor:
        if self.fit_intercept:
            x = torch.cat([torch.ones(x.shape[0], 1), x], dim=1)
        return x @ self.weights


class LinUCB(Bandit):
    DEFAULT_CONFIG = {}

    def __init__(self, action_space: Space, obs_space: Space, **kwargs):
        super().__init__(action_space, obs_space, **kwargs)
        self.alpha = torch.zeros(self.n_arms)
        d = self.obs_space.shape
        self.A = [torch.eye(d) for _ in range(self.n_arms)]
        # self.b = [torch.zeros([d, 1]) for _ in range(self.n_arms)]
        self.b = torch.zeros([self.n_arms, d, 1])
        # self.theta = [torch.zeros([d, 1]) for _ in range(self.n_arms)]
        self.ridge = RidgeRegressor(self.n_arms)

    def act(self, obs: Tensor) -> Tensor:
        probs = []
        for i, (A, b, alpha) in enumerate(zip(self.A, self.b, self.alpha)):
            A_inv = torch.linalg.pinv(A)
            theta = torch.dot(A_inv, b)
            p = torch.dot(theta.T, obs) + alpha * torch.sqrt(torch.dot(obs.T, torch.dot(A_inv, obs)))
            probs.append(p)
        return torch.tensor(probs).argmax()

    def update(self, batch: List[Transition]) -> dict:
        return super().update(batch)
