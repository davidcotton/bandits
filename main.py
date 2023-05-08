import argparse
from datetime import datetime

import mlflow

from bandits.bandits import QBandit
from bandits.bandits.exp3_bandit import Exp3Bandit
from bandits.envs import BernoulliEnv
from bandits.runner import Runner
from bandits.samplers import EpsilonGreedySampler


POLICIES = {
    # 'random': (RandomSampler, {}),
    'epsilon greedy (10%)': (EpsilonGreedySampler, {'epsilon': 0.1}),
    # # 'epsilon greedy (5%)': (EpsilonGreedySampler, {'epsilon': 0.05}),
    'epsilon greedy (1%)': (EpsilonGreedySampler, {'epsilon': 0.01}),
    # # 'epsilon weighted (alpha 0.1)': (EpsilonGreedyWeightedSampler, {'epsilon': 0.1, 'alpha': 0.1}),
    # # 'epsilon weighted (alpha 0.01)': (EpsilonGreedyWeightedSampler, {'epsilon': 0.1, 'alpha': 0.01}),
    # # 'softmax (0.01)': (SoftmaxSampler, {'temperature': 0.01}),
    # 'softmax (0.1)': (SoftmaxSampler, {'temperature': 0.1}),
    # # 'softmax (0.2)': (SoftmaxSampler, {'temperature': 0.2}),
    # # 'softmax (0.3)': (SoftmaxSampler, {'temperature': 0.3}),
    # 'softmax (0.4)': (SoftmaxSampler, {'temperature': 0.4}),
    # # 'softmax (0.5)': (SoftmaxSampler, {'temperature': 0.5}),
    # # 'softmax (0.6)': (SoftmaxSampler, {'temperature': 0.6}),
    # # 'softmax (0.7)': (SoftmaxSampler, {'temperature': 0.7}),
    # # 'softmax (0.8)': (SoftmaxSampler, {'temperature': 0.8}),
    # # 'softmax (0.9)': (SoftmaxSampler, {'temperature': 0.9}),
    # 'ucb1': (UCB1Sampler, {}),
    # 'exp3': (Exp3Bandit, {}),
    # 'exp3 (gamma=0.01)': (Exp3Bandit, {'gamma': 0.01}),
    'exp3 (gamma=0.05)': (Exp3Bandit, {'gamma': 0.05}),
    'exp3 (gamma=0.07)': (Exp3Bandit, {'gamma': 0.07}),
    'exp3 (gamma=0.1)': (Exp3Bandit, {'gamma': 0.1}),
    'exp3 (gamma=0.2)': (Exp3Bandit, {'gamma': 0.2}),
    'exp3 (gamma=0.3)': (Exp3Bandit, {'gamma': 0.3}),
    # 'exp3 (gamma=0.4)': (Exp3Bandit, {'gamma': 0.4}),
    # 'exp3 (gamma=0.5)': (Exp3Bandit, {'gamma': 0.5}),
}
DEFAULT_MLFLOW_SERVER_URI = "http://localhost:5000"


def main(args):
    env = BernoulliEnv(args.n_arms)
    bandits = {
        "simple_q_bandit": QBandit(),
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
    parser.add_argument('--n-steps', type=int, default=10, help='The number of steps to train on')
    parser.add_argument('--n-trials', type=int, default=5, help='The number of trials to run for each algorithm')
    parser.add_argument(
        "--mlflow_server_uri",
        default=DEFAULT_MLFLOW_SERVER_URI,
        help="The URI of the MLFlow Tracking Server to use.",
    )
    main(parser.parse_args())
