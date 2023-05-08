import argparse
from datetime import datetime

import mlflow

from bandits.bandits import SimpleQBandit
from bandits.envs import BernoulliEnv
from bandits.runner import Runner
from bandits.selectors import EpsilonGreedySelector, Exp3Selector


POLICIES = {
    # 'random': (RandomSelector, {}),
    'epsilon greedy (10%)': (EpsilonGreedySelector, {'epsilon': 0.1}),
    # # 'epsilon greedy (5%)': (EpsilonGreedySelector, {'epsilon': 0.05}),
    'epsilon greedy (1%)': (EpsilonGreedySelector, {'epsilon': 0.01}),
    # # 'epsilon weighted (alpha 0.1)': (EpsilonGreedyWeightedSelector, {'epsilon': 0.1, 'alpha': 0.1}),
    # # 'epsilon weighted (alpha 0.01)': (EpsilonGreedyWeightedSelector, {'epsilon': 0.1, 'alpha': 0.01}),
    # # 'softmax (0.01)': (SoftmaxSelector, {'temperature': 0.01}),
    # 'softmax (0.1)': (SoftmaxSelector, {'temperature': 0.1}),
    # # 'softmax (0.2)': (SoftmaxSelector, {'temperature': 0.2}),
    # # 'softmax (0.3)': (SoftmaxSelector, {'temperature': 0.3}),
    # 'softmax (0.4)': (SoftmaxSelector, {'temperature': 0.4}),
    # # 'softmax (0.5)': (SoftmaxSelector, {'temperature': 0.5}),
    # # 'softmax (0.6)': (SoftmaxSelector, {'temperature': 0.6}),
    # # 'softmax (0.7)': (SoftmaxSelector, {'temperature': 0.7}),
    # # 'softmax (0.8)': (SoftmaxSelector, {'temperature': 0.8}),
    # # 'softmax (0.9)': (SoftmaxSelector, {'temperature': 0.9}),
    # 'ucb1': (UCB1Selector, {}),
    # 'exp3': (Exp3Selector, {}),
    # 'exp3 (gamma=0.01)': (Exp3Selector, {'gamma': 0.01}),
    'exp3 (gamma=0.05)': (Exp3Selector, {'gamma': 0.05}),
    'exp3 (gamma=0.07)': (Exp3Selector, {'gamma': 0.07}),
    'exp3 (gamma=0.1)': (Exp3Selector, {'gamma': 0.1}),
    'exp3 (gamma=0.2)': (Exp3Selector, {'gamma': 0.2}),
    'exp3 (gamma=0.3)': (Exp3Selector, {'gamma': 0.3}),
    # 'exp3 (gamma=0.4)': (Exp3Selector, {'gamma': 0.4}),
    # 'exp3 (gamma=0.5)': (Exp3Selector, {'gamma': 0.5}),
}
DEFAULT_MLFLOW_SERVER_URI = "http://localhost:5000"


def main(args):
    env = BernoulliEnv(args.n_arms)
    bandits = {
        "simple_q_bandit": SimpleQBandit(),
    }
    runner = Runner(env, bandits, args.n_steps)

    # run experiment
    # mlflow.set_tracking_uri(args.mlflow_server_uri)
    # experiment_name = f"{args.experiment}"
    # mlflow.set_experiment(experiment_name)
    results = runner.run()
    if "summary" in results:
        summary = results["summary"]
        # filename = f"{datetime.today().strftime('%Y-%m-%d-%H%M')}_{experiment_name}.csv"
        # summary.to_csv(get_summaries_dir() / filename)
        print("\n", summary.to_markdown())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['bernoulli', 'gaussian'], default='bernoulli')
    parser.add_argument("--n-arms", "-n", type=int, default=10)
    parser.add_argument('--n-steps', type=int, default=1000, help='The number of steps to train on')
    parser.add_argument('--n-trials', type=int, default=5, help='The number of trials to run for each algorithm')
    parser.add_argument(
        "--mlflow_server_uri",
        default=DEFAULT_MLFLOW_SERVER_URI,
        help="The URI of the MLFlow Tracking Server to use.",
    )
    main(parser.parse_args())
