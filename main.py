import argparse
from datetime import datetime

import mlflow

from bandits.bandits import build_bandits
from bandits.envs import build_env
from bandits.measures import build_measures
from bandits.runner import Runner
from bandits.utils import get_summaries_dir, load_config

DEFAULT_MLFLOW_SERVER_URI = "http://127.0.0.1:5000"


def main(args):
    config = load_config(args.experiment)
    required_config_keys = ("global", "runner", "environment", "models", "measures")
    for k in required_config_keys:
        config.setdefault(k, {})

    env = build_env(config)
    bandits = build_bandits(config, env)
    measures = build_measures(config)
    runner = Runner(config, env, bandits, measures)

    # run experiment
    mlflow.set_tracking_uri(args.mlflow_server_uri)
    experiment_name = f"{args.experiment}"
    mlflow.set_experiment(experiment_name)
    results = runner.run()
    if "summary" in results:
        summary = results["summary"]
        filename = f"{datetime.today().strftime('%Y-%m-%d-%H%M')}_{experiment_name}.csv"
        summary.to_csv(get_summaries_dir() / filename)
        print("\n", summary.to_markdown())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="The name of the experiment configuration file to use."
    )
    parser.add_argument(
        "--mlflow_server_uri",
        default=DEFAULT_MLFLOW_SERVER_URI,
        help="The URI of the MLFlow Tracking Server to use.",
    )
    main(parser.parse_args())
