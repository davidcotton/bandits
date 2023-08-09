from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd
from torch import Tensor
from tqdm import tqdm

from bandits.bandits import Bandits
from bandits.envs import Env
from bandits.measures import Measures
from bandits.transition import Transition


class Runner:
    def __init__(self, config: dict, env: Env, bandits: Bandits, measures: Measures) -> None:
        super().__init__()
        self.config = config
        self.n_steps = int(config["global"]["n_steps"])
        self.env: Env = env
        self.bandits: Bandits = bandits
        self.measures: Measures = measures

    def run(self) -> dict:
        results = []
        for bandit_name, bandit in tqdm(self.bandits.items()):
            metrics = defaultdict(list)
            run_params = {
                "bandit": bandit_name,
                "env": self.env.__class__.__name__,
                **self.config["global"],
            }
            with mlflow.start_run() as run:
                mlflow.log_params(run_params)
                obs, terminal = self.env.reset()
                for _ in tqdm(range(self.n_steps), mininterval=10):
                    actions = bandit.act(obs.unsqueeze(0)).squeeze(1)
                    # actions = bandit.act(obs.unsqueeze(0))
                    next_obs, rewards, terminal, info = self.env.step(actions)
                    batch = [Transition(obs, actions, next_obs, rewards, terminal)]
                    step_metrics = bandit.update(batch)
                    metrics["reward"].append(rewards.sum().item())
                    for k, v in info.items():
                        if isinstance(v, Tensor):
                            v = v.mean().item()
                        metrics[k].append(v)
                    for k, v in step_metrics.items():
                        metrics[k].append(v)
                    obs = next_obs

                run_metrics = {}
                for k, v in metrics.items():
                    run_metrics[f"train_{k}_mean"] = np.mean(v).item()
                    run_metrics[f"train_{k}_sum"] = np.sum(v).item()
                    run_metrics[f"train_{k}_std"] = np.std(v).item()
                mlflow.log_metrics(run_metrics)
                results.append({**run_metrics, **run_params})

        summary = (
            pd.DataFrame.from_records(results)
            # .sort_values("test_k_precision", ascending=False)
        )

        return {
            "results": results,
            "summary": summary,
        }
