import os
import argparse

import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
import json_tricks

from spatial_validation.data import SpatialDataset, Dataset
from spatial_validation.data.data_utils import ensure_numpy_dict, dataset_to_dict


def generate_test_sites(num_sites: int = 100):
    return np.linspace(0, 1, num_sites)


def generate_validation_sites(num_sites: int = 100):
    return (np.random.rand(num_sites)) ** (1 / 2)


def generate_train_sites(num_sites: int = 100):
    return (np.random.rand(num_sites)) ** (1 / 2)


def generate_sites(num_train, num_val, num_test):
    return np.concatenate(
        [
            generate_train_sites(num_train),
            generate_validation_sites(num_val),
            generate_test_sites(num_test),
        ],
        axis=0,
    )


def generate_covariates():
    return None


def generate_response(sites: ArrayLike) -> ArrayLike:
    n = len(sites)
    noise = 0.02 * np.random.rand(n)
    return np.abs(sites - 0.5) + noise[:, None]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nv",
        "--num_val",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-nt",
        "--num_train",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    n_train = args.num_train
    n_val=args.num_val
    n_test = 21

    datadir = Path(Path(__file__).parent, "data", f"nval_{n_val}")
    os.makedirs(str(datadir), exist_ok=True)


    def generate_data(seed: int):
        np.random.seed(seed)
        allsites = generate_sites(n_train, n_val, n_test)[:, None]
        response = generate_response(allsites)

        data_fp = str(Path(datadir, f"seed-{seed}.json"))
        training_data = SpatialDataset(
            S=allsites[:n_train], X=None, Y=response[:n_train]
        )
        validation_data = SpatialDataset(
            S=allsites[n_train : n_val + n_train],
            X=None,
            Y=response[n_train : n_val + n_train],
        )
        test_data = SpatialDataset(
            S=allsites[n_val + n_train :],
            X=None,
            Y=response[n_val + n_train :],
        )
        dataset = Dataset(training_data, validation_data, test_data)
        data_dict = dataset_to_dict(dataset)
        data_dict = ensure_numpy_dict(data_dict)
        json_tricks.dump(data_dict, data_fp)

    for seed in np.arange(100):
        generate_data(seed)
