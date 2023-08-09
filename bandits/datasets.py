import zipfile
from pathlib import Path
from urllib.request import (
    urlcleanup,
    urlretrieve,
)

import pandas as pd

DATA_DIR = Path.home() / ".bandits_data"
DATA_DIR.mkdir(exist_ok=True)
DATASETS = {
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
}


class Dataset:
    @staticmethod
    def download_dataset(dataset_name) -> None:
        dataset_url = DATASETS[dataset_name]
        tmp_file_path, headers = urlretrieve(dataset_url)
        with zipfile.ZipFile(tmp_file_path, "r") as zip_file_handle:
            zip_file_handle.extractall(DATA_DIR)
        urlcleanup()


class MovieLensDataset(Dataset):
    def __init__(self, dataset_name="ml-100k") -> None:
        self.dataset_file_path = DATA_DIR / dataset_name
        if not self.dataset_file_path.is_dir():
            self.download_dataset(dataset_name)

    def __call__(self, *args, **kwargs):
        users = pd.read_csv(
            self.dataset_file_path / "u.user",
            sep="|",
            header=None,
            names=["user_id", "age", "gender", "occupation", "zip_code"],
        )
        ratings = pd.read_csv(
            self.dataset_file_path / "u.data",
            sep="\t",
            header=None,
            names=["user_id", "movie_id", "rating", "unix_timestamp"],
            converters={
                "unix_timestamp": lambda ts: pd.Timestamp(int(ts), unit="s").to_datetime64(),
            }
        )
        return ratings.merge(users, how="left", on="user_id")
