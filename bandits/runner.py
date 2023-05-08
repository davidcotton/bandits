from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

from bandits.bandits import Bandits
from bandits.envs import Env
from bandits.transition import Transition


class Runner:
    def __init__(self, env: Env, bandits: Bandits, n_steps: int) -> None:
        super().__init__()
        self.env: Env = env
        self.bandits: Bandits = bandits
        self.n_steps = int(n_steps)

    def run(self) -> dict:
        results = []
        for bandit_name, bandit in tqdm(self.bandits.items()):
            run_params = {
                "bandit": bandit_name,
            }
            metrics = defaultdict(list)
            with mlflow.start_run() as run:
                obs, terminal = self.env.reset()
                for _ in tqdm(range(self.n_steps)):
                    actions = bandit.act(obs.unsqueeze(0)).squeeze()
                    next_obs, rewards, terminal, info = self.env.step(actions)
                    batch = [Transition(obs, actions, next_obs, rewards, terminal)]
                    step_metrics = bandit.update(batch)
                    metrics["reward"].append(rewards.sum().item())
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


# class Agent:
#     """Wrapper around the selector and bandits."""
#
#     def __init__(self, bandits: List[Bandit], selector: Sampler, debug=False) -> None:
#         super().__init__()
#         self.bandits: List[Bandit] = bandits
#         self.selector: Sampler = selector
#         self.debug = debug
#
#     def train(self, nb_steps: int) -> dict:
#         """
#         Train the agent for a number of steps.
#
#         :param nb_steps: the number of steps to train for
#         :return: the training history
#         """
#         print(f'Training {self.selector.__class__.__name__} for {nb_steps:,} steps')
#         results = {
#             'avg_reward': [],
#             'avg_regret': [],
#             'params': {
#                 'total_rewards': {},
#                 'avg_rewards': {},
#                 'weighted_q': {},
#             }
#         }
#         if self.debug:
#             for key in results['params'].keys():
#                 for i in range(len(self.bandits)):
#                     results['params'][key][i] = []
#
#         optimal_bandit = max(bandit for bandit in self.bandits)
#         avg_reward = 0.0
#         avg_regret = 0.0
#
#         for n in range(1, nb_steps + 1):
#             reward, bandit_idx = self.act()
#             avg_reward = self.incremental_mean(avg_reward, reward, n)
#             results['avg_reward'].append(avg_reward)
#
#             # measure regret
#             if self.bandits[0].is_nonstationary:  # if bandits aren't stationary we'll need to keep checking this
#                 optimal_bandit = max(bandit for bandit in self.bandits)
#             regret = optimal_bandit.p - self.bandits[bandit_idx].p
#             avg_regret = self.incremental_mean(avg_regret, regret, n)
#             results['avg_regret'].append(avg_regret)
#
#             if self.debug:
#                 try:
#                     for i, value in enumerate(self.selector.total_rewards):
#                         results['params']['total_rewards'][i].append(value)
#                     for i, value in enumerate(self.selector.avg_rewards):
#                         results['params']['avg_rewards'][i].append(value)
#                     for i, value in enumerate(self.selector.q):
#                         results['params']['weighted_q'][i].append(value)
#                 except NameError:
#                     pass
#
#         return results
#
#     @staticmethod
#     def incremental_mean(mean, value, n):
#         return mean + (value - mean) / n
#
#     def act(self) -> Tuple[float, int]:
#         """
#         Select a bandit with our selector, take the action and receive the reward.
#         :return: a tuple of the reward received and the bandit arm chosen
#         """
#         bandit_idx = self.selector.select_action()
#         reward = self.bandits[bandit_idx].pull()
#         self.selector.update(bandit_idx, reward)
#         return reward, bandit_idx
#
#     def print_estimates(self) -> None:
#         print(' Real || Pred || Diff')
#         for real, predicted in zip(self.bandits, self.selector.q):
#             real = real.p
#             diff = abs(real - predicted)
#             print(f' {real:0.3f} | {predicted:0.3f} | {diff:0.3f}')
#
#
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

