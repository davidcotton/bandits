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
            run_params = {
                "bandit": bandit_name,
                **self.config["global"]
            }
            metrics = defaultdict(list)
            with mlflow.start_run() as run:
                obs, terminal = self.env.reset()
                for _ in tqdm(range(self.n_steps)):
                    actions = bandit.act(obs.unsqueeze(0)).squeeze(1)
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
                mlflow.log_params(run_params)
                results.append({**run_metrics, **run_params})

        summary = (
            pd.DataFrame.from_records(results)
            # .sort_values("test_k_precision", ascending=False)
        )

        return {
            "results": results,
            "summary": summary,
        }


# def print_bandit_summary(bandits: List[Bandit]) -> None:
#     """Print a summary of the bandits randomly generated."""
#     print()
#     for i, bandit in enumerate(bandits):
#         print(f'{i}) {bandit}')
#     best = max(bandit for bandit in bandits)
#     print('\nBest: {}) {}\n\n#########################\n'.format(bandits.index(best), str(best)))
#
#
# def plot_results(history, bandit_cls, nb_bandits) -> None:
#     """Plot the average rewards and average regret for each algorithm."""
#
#     subplots = {
#         'avg_rewards': {},
#         'avg_regret': {},
#     }
#     for policy_name, policy_results in history.items():
#         avg_rewards = [r['avg_reward'] for r in policy_results]
#         avg_rewards = list(map(np.mean, zip(*avg_rewards)))
#         subplots['avg_rewards'][policy_name] = avg_rewards
#
#         avg_regret = [r['avg_regret'] for r in policy_results]
#         avg_regret = list(map(np.mean, zip(*avg_regret)))
#         subplots['avg_regret'][policy_name] = avg_regret
#
#     fig, ax = plt.subplots(nrows=len(subplots), ncols=1)
#     fig.suptitle(f'{bandit_cls.__name__}x{nb_bandits}')
#     for row, subplot_name in zip(ax, subplots):
#         row.set_ylabel(subplot_name)
#         subplot_values = subplots[subplot_name]
#         for key, value in subplot_values.items():
#             row.plot(value, label=key)
#         row.legend()
#         # if len(value) > 500:  # make it easier to read when many steps
#         #     row.set_xscale('log')
#         row.grid()
#
#     plt.xlabel('Iterations')
#     fig.subplots_adjust(top=0.88)
#     plt.show()
#
#
# def plot_debug(history, bandit_cls, nb_bandits) -> None:
#     subplots = {
#         # 'total_rewards': history['ucb1'][0]['params']['total_rewards'],
#         'avg_rewards': history['ucb1'][0]['params']['avg_rewards'],
#         'weighted_q': history['ucb1'][0]['params']['weighted_q'],
#     }
#
#     fig, ax = plt.subplots(nrows=len(subplots), ncols=1)
#     fig.suptitle(f'{bandit_cls.__name__}x{nb_bandits}')
#     for row, subplot_name in zip(ax, subplots):
#         row.set_ylabel(subplot_name)
#         subplot_values = subplots[subplot_name]
#         for key, value in subplot_values.items():
#             row.plot(value, label=key)
#         row.legend()
#         if len(value) > 500:  # make it easier to read when many steps
#             row.set_xscale('log')
#         row.grid()
#
#     plt.xlabel('Iterations')
#     fig.subplots_adjust(top=0.88)
#     plt.show()

