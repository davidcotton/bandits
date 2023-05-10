__all__ = [
    "Measures",
    "build_measures",
]

from typing import Callable, Mapping

Measures = Mapping[str, Callable]


def build_measures(config):
    measures_config = config["measures"]
    measures = {}
    for name, cfg in measures_config.items():
        measures[name] = cfg
    return measures
