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
            self.weights, _, _, _ = torch.linalg.lstsq(rhs, lhs)
        else:
            ridge = self.alpha * torch.eye(lhs.shape[0])
            self.weights, _, _, _ = torch.linalg.lstsq(rhs, lhs + ridge)

    def predict(self, x: Tensor) -> Tensor:
        if self.fit_intercept:
            x = torch.cat([torch.ones(x.shape[0], 1), x], dim=1)
        return x @ self.weights


class LinUCB(Bandit):
    DEFAULT_CONFIG = {
        "alpha": 1.0,
    }

    def __init__(self, action_space: Space, obs_space: Space, **kwargs):
        super().__init__(action_space, obs_space, **kwargs)
        self.config.update({
            **LinUCB.DEFAULT_CONFIG,  # self config
            **self.DEFAULT_CONFIG,  # child config
            **kwargs,  # override config
        })
        # d: context dimensionality
        d = torch.prod(torch.tensor(self.obs_space.shape)).item()
        # A_k = (d * d) identity matrix
        self.A = [torch.eye(d) for _ in range(self.n_arms)]
        # b: (d * 1) corresponding response vector
        # Equal to D_a.T * c_a in ridge regression formula
        # self.b = [torch.zeros([d, 1]) for _ in range(self.n_arms)]
        self.b = torch.zeros([self.n_arms, d, 1])
        # self.theta = [torch.zeros([d, 1]) for _ in range(self.n_arms)]
        self.ridge = RidgeRegressor(self.n_arms)

    def act(self, obs: Tensor) -> Tensor:
        # Reshape covariates input into (d x 1) shape vector
        obs = obs.T
        probs = []
        for i, (A, b) in enumerate(zip(self.A, self.b)):
            A_inv = torch.linalg.pinv(A)
            theta = A_inv.mm(b)
            p = theta.T.mm(obs) + self.alpha * torch.sqrt(obs.T.mm(A_inv.mm(obs)))
            probs.append(p)
        # return torch.tensor(probs).argmax()
        probs = torch.tensor(probs)
        best_action = probs.argmax()
        best_action = best_action.unsqueeze(0).unsqueeze(0)
        return best_action

    def update(self, batch: List[Transition]) -> dict:
        metrics = super().update(batch)
        for obs, actions, _, rewards, _ in batch:
            obs = obs.unsqueeze(-1)  # reshape covariates input into (d x 1)
            for a, r in zip(actions, rewards):
                self.A[a] += obs.mm(obs.T)
                self.b[a] += r * obs
        return metrics

    @property
    def alpha(self):
        return float(self.config["alpha"])
