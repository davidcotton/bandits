import argparse
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.bandits import Bandit, BernoulliBandit, GaussianBandit
from src.policies import Policy, EpsilonGreedyPolicy, Exp3Policy


DEBUG = True
BANDITS = {
    'bernoulli': BernoulliBandit,
    'gaussian': GaussianBandit,
}
POLICIES = {
    # 'random': (RandomPolicy, {}),
    'epsilon greedy (10%)': (EpsilonGreedyPolicy, {'epsilon': 0.1}),
    # # 'epsilon greedy (5%)': (EpsilonGreedyPolicy, {'epsilon': 0.05}),
    'epsilon greedy (1%)': (EpsilonGreedyPolicy, {'epsilon': 0.01}),
    # # 'epsilon weighted (alpha 0.1)': (EpsilonGreedyWeightedPolicy, {'epsilon': 0.1, 'alpha': 0.1}),
    # # 'epsilon weighted (alpha 0.01)': (EpsilonGreedyWeightedPolicy, {'epsilon': 0.1, 'alpha': 0.01}),
    # # 'softmax (0.01)': (SoftmaxPolicy, {'temperature': 0.01}),
    # 'softmax (0.1)': (SoftmaxPolicy, {'temperature': 0.1}),
    # # 'softmax (0.2)': (SoftmaxPolicy, {'temperature': 0.2}),
    # # 'softmax (0.3)': (SoftmaxPolicy, {'temperature': 0.3}),
    # 'softmax (0.4)': (SoftmaxPolicy, {'temperature': 0.4}),
    # # 'softmax (0.5)': (SoftmaxPolicy, {'temperature': 0.5}),
    # # 'softmax (0.6)': (SoftmaxPolicy, {'temperature': 0.6}),
    # # 'softmax (0.7)': (SoftmaxPolicy, {'temperature': 0.7}),
    # # 'softmax (0.8)': (SoftmaxPolicy, {'temperature': 0.8}),
    # # 'softmax (0.9)': (SoftmaxPolicy, {'temperature': 0.9}),
    # 'ucb1': (UCB1Policy, {}),
    # 'exp3': (Exp3Policy, {}),
    # 'exp3 (gamma=0.01)': (Exp3Policy, {'gamma': 0.01}),
    'exp3 (gamma=0.05)': (Exp3Policy, {'gamma': 0.05}),
    'exp3 (gamma=0.07)': (Exp3Policy, {'gamma': 0.07}),
    'exp3 (gamma=0.1)': (Exp3Policy, {'gamma': 0.1}),
    'exp3 (gamma=0.2)': (Exp3Policy, {'gamma': 0.2}),
    'exp3 (gamma=0.3)': (Exp3Policy, {'gamma': 0.3}),
    # 'exp3 (gamma=0.4)': (Exp3Policy, {'gamma': 0.4}),
    # 'exp3 (gamma=0.5)': (Exp3Policy, {'gamma': 0.5}),
}


class Agent:
    """Wrapper around the policy and bandits."""

    def __init__(self, bandits: List[Bandit], policy: Policy, debug=False) -> None:
        super().__init__()
        self.bandits: List[Bandit] = bandits
        self.policy: Policy = policy
        self.debug = debug

    def train(self, nb_steps: int) -> dict:
        """
        Train the agent for a number of steps.

        :param nb_steps: the number of steps to train for
        :return: the training history
        """
        print(f'Training {self.policy.__class__.__name__} for {nb_steps:,} steps')
        results = {
            'avg_reward': [],
            'avg_regret': [],
            'params': {
                'total_rewards': {},
                'avg_rewards': {},
                'weighted_q': {},
            }
        }
        if self.debug:
            for key in results['params'].keys():
                for i in range(len(self.bandits)):
                    results['params'][key][i] = []

        optimal_bandit = max(bandit for bandit in self.bandits)
        avg_reward = 0.0
        avg_regret = 0.0

        for n in range(1, nb_steps + 1):
            reward, bandit_idx = self.act()
            avg_reward = self.incremental_mean(avg_reward, reward, n)
            results['avg_reward'].append(avg_reward)

            # measure regret
            if self.bandits[0].is_nonstationary:  # if bandits aren't stationary we'll need to keep checking this
                optimal_bandit = max(bandit for bandit in self.bandits)
            regret = optimal_bandit.p - self.bandits[bandit_idx].p
            avg_regret = self.incremental_mean(avg_regret, regret, n)
            results['avg_regret'].append(avg_regret)

            if self.debug:
                try:
                    for i, value in enumerate(self.policy.total_rewards):
                        results['params']['total_rewards'][i].append(value)
                    for i, value in enumerate(self.policy.avg_rewards):
                        results['params']['avg_rewards'][i].append(value)
                    for i, value in enumerate(self.policy.q):
                        results['params']['weighted_q'][i].append(value)
                except NameError:
                    pass

        return results

    @staticmethod
    def incremental_mean(mean, value, n):
        return mean + (value - mean) / n

    def act(self) -> Tuple[float, int]:
        """
        Select a bandit with our policy, take the action and receive the reward.

        :return: a tuple of the reward received and the bandit arm chosen
        """
        bandit_idx = self.policy.select_action()
        reward = self.bandits[bandit_idx].pull()
        self.policy.update(bandit_idx, reward)
        return reward, bandit_idx

    def print_estimates(self) -> None:
        print(' Real || Pred || Diff')
        for real, predicted in zip(self.bandits, self.policy.q):
            real = real.p
            diff = abs(real - predicted)
            print(f' {real:0.3f} | {predicted:0.3f} | {diff:0.3f}')


def print_bandit_summary(bandits: List[Bandit]) -> None:
    """Print a summary of the bandits randomly generated."""
    print()
    for i, bandit in enumerate(bandits):
        print(f'{i}) {bandit}')
    best = max(bandit for bandit in bandits)
    print('\nBest: {}) {}\n\n#########################\n'.format(bandits.index(best), str(best)))


def plot_results(history, bandit_cls, nb_bandits) -> None:
    """Plot the average rewards and average regret for each algorithm."""

    subplots = {
        'avg_rewards': {},
        'avg_regret': {},
    }
    for policy_name, policy_results in history.items():
        avg_rewards = [r['avg_reward'] for r in policy_results]
        avg_rewards = list(map(np.mean, zip(*avg_rewards)))
        subplots['avg_rewards'][policy_name] = avg_rewards

        avg_regret = [r['avg_regret'] for r in policy_results]
        avg_regret = list(map(np.mean, zip(*avg_regret)))
        subplots['avg_regret'][policy_name] = avg_regret

    fig, ax = plt.subplots(nrows=len(subplots), ncols=1)
    fig.suptitle(f'{bandit_cls.__name__}x{nb_bandits}')
    for row, subplot_name in zip(ax, subplots):
        row.set_ylabel(subplot_name)
        subplot_values = subplots[subplot_name]
        for key, value in subplot_values.items():
            row.plot(value, label=key)
        row.legend()
        # if len(value) > 500:  # make it easier to read when many steps
        #     row.set_xscale('log')
        row.grid()

    plt.xlabel('Iterations')
    fig.subplots_adjust(top=0.88)
    plt.show()


def plot_debug(history, bandit_cls, nb_bandits) -> None:
    subplots = {
        # 'total_rewards': history['ucb1'][0]['params']['total_rewards'],
        'avg_rewards': history['ucb1'][0]['params']['avg_rewards'],
        'weighted_q': history['ucb1'][0]['params']['weighted_q'],
    }

    fig, ax = plt.subplots(nrows=len(subplots), ncols=1)
    fig.suptitle(f'{bandit_cls.__name__}x{nb_bandits}')
    for row, subplot_name in zip(ax, subplots):
        row.set_ylabel(subplot_name)
        subplot_values = subplots[subplot_name]
        for key, value in subplot_values.items():
            row.plot(value, label=key)
        row.legend()
        if len(value) > 500:  # make it easier to read when many steps
            row.set_xscale('log')
        row.grid()

    plt.xlabel('Iterations')
    fig.subplots_adjust(top=0.88)
    plt.show()


def main(args):
    # create some bandits
    bandit_cls = BANDITS[args.bandit_type]
    num_bandits = args.num_bandits
    bandits = bandit_cls.build(num_bandits)
    if len(bandits) <= 10:
        print_bandit_summary(bandits)

    # for each policy
    #   - train an agent with that policy for a number of trials
    results = {}
    for name, policy_parts in POLICIES.items():
        policy_cls, kwargs = policy_parts
        policy = policy_cls(len(bandits), debug=DEBUG, **kwargs)
        agent = Agent(bandits, policy, debug=DEBUG)
        policy_results = []
        for _ in range(args.trials):
            policy_results.append(agent.train(args.steps))
        results[name] = policy_results
    plot_results(results, bandit_cls, num_bandits)
    if DEBUG:
        plot_debug(results, bandit_cls, num_bandits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-bandits', type=int, default=10)
    parser.add_argument('--bandit-type', choices=['bernoulli', 'gaussian'], default='bernoulli')
    parser.add_argument('--steps', type=int, default=1000, help='The number of steps to train on')
    parser.add_argument('--trials', type=int, default=5, help='The number of trials to run for each algorithm')
    main(parser.parse_args())
