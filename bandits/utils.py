from pathlib import Path
from typing import Tuple

import yaml

DIRS = {
    "configs": "configs",
    "data": "artifacts/data",
    "imgs": "artifacts/imgs",
    "summaries": "artifacts/summaries",
}


def get_project_root() -> Path:
    """Define the project root in relation to this utils file.
    By using this method, any entrypoint, such as ipynbs in any dir,
    can use project assets."""
    return Path(__file__).parent.parent


def _get_dir(dir_name) -> Path:
    path = Path(get_project_root(), *dir_name.split("/"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    return _get_dir(DIRS["data"])


def get_imgs_dir() -> Path:
    return _get_dir(DIRS["imgs"])


def get_summaries_dir() -> Path:
    return _get_dir(DIRS["summaries"])


def load_config(experiment_name: str) -> dict:
    config_filename = _get_dir(DIRS["configs"]) / f"{experiment_name}.yml"
    with open(config_filename, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    return config


def extract_module(module_str: str, default_namespace=None) -> Tuple[str, str]:
    parts = module_str.rsplit(".", 1)
    if len(parts) == 2:
        namespace, import_ = parts
    elif default_namespace is not None:
        import_ = parts[0]
        namespace = default_namespace
    else:
        raise ValueError("Default module namespace not specified")
    return namespace, import_
