import argparse
import os
import sys
from pathlib import Path
from typing import Dict
from joblib import Parallel, delayed

import json_tricks
import numpy as np
from gpflow.kernels import Matern32, RBF, Matern12

from spatial_validation.data import ensure_numpy_dict, load_dataset_from_json
from spatial_validation.experiment import Experiment
from spatial_validation.losses import SquaredLoss, TruncatedSquaredLoss
from spatial_validation.models import KRRSpatialRegression

def build_kernel(
    num_covariates: int,
    spatial_dimension: int,
    params: Dict,
    kernel_name: str,
):
    if kernel_name == "rbf":
        kernel_fn = RBF
    elif kernel_name == "matern32":
        kernel_fn = Matern32
    elif kernel_name == "matern12":
        kernel_fn = Matern12
    else:
        raise NotImplementedError

    response_kernel = kernel_fn(
        lengthscales=params["response_lengthscale"],
        active_dims=np.arange(
            num_covariates - 1
        ),  # We only use first covariate, this makes things mis-specified
    )
    spatial_noise_kernel = kernel_fn(
        lengthscales=params["noise_lengthscale"],
        variance=params["spatial_noise_variance"],
        active_dims=np.arange(num_covariates, num_covariates + spatial_dimension),
    )

    kernel = response_kernel + spatial_noise_kernel

    return kernel


def build_experiment_args(
    seed: int, datadir: Path, kernel_name: str = "matern32"
) -> Dict:
    datapath = Path(datadir, f"seed-{seed}.json")
    dataset = load_dataset_from_json(str(datapath))

    loss = TruncatedSquaredLoss(max_value=1.0)
    # Estimators to Use
    estimator_names = [
        "BasicLossEstimator",
        "NearestNeighborEstimator",
        "FitNearestNeighborEstimator",
    ]
    estimator_parameters = [dict(), dict(), dict()]
    # Models and parameters
    regularization_parameter = 0.1

    model_names = ["KRRSpatialRegression"]
    kernel = build_kernel(
        num_covariates=2,
        spatial_dimension=2,
        params=data_params,
        kernel_name=kernel_name,
    )
    model_parameters = [
        dict(
            response_kernel=kernel,
            regularization_parameter=regularization_parameter,
        )
    ]
    return (
        dict(
            dataset=dataset,
            model_names=model_names,
            model_parameters=model_parameters,
            param_names=[""],
            estimator_names=estimator_names,
            estimator_parameters=estimator_parameters,
            loss=loss,
        ),
        datapath,
    )


if __name__ == "__main__":
    # add argparse to choose dataset
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cluster",
        choices=["cluster", "point_prediction"],
    )
    parser.add_argument(
        "-nv",
        "--num_val",
        type=int,
        default=500,
    )
    parser.add_argument(
        "-nt",
        "--num_train",
        type=int,
        default=1000,
    )
    # add argparse to choose which seeds to run
    parser.add_argument("-t", "--threads", type=int, default=5)
    parser.add_argument("-k", "--kernel", type=str, default="matern32", choices=["rbf", "matern32", "matern12"])
    # parse args
    args = parser.parse_args()
    # Load Data
    root_data_dir = Path(
        Path(__file__).parent.parent.parent,
        "spatial_validation",
        "data",
        "synthetic",
        "data",
    )
    root_results_dir = Path(Path(__file__).parent, "results")
    if args.dataset == "cluster":
        ds_name = "grid"
        train_dist = "Cluster"
    elif args.dataset == "point_prediction":
        ds_name = "point"
        train_dist = "Uniform"
    else:
        raise ValueError("Invalid dataset choice")
    model_name = "gp"
    kernel_name = args.kernel
    datadir = Path(
        root_data_dir,
        f"{ds_name}-{model_name}",
        f"{train_dist}-{args.num_train}Train_{args.num_val}Val",
    )
    resultsdir = Path(
        root_results_dir,
        f"{ds_name}-{model_name}",
        f"{train_dist}-{args.num_train}Train_{args.num_val}Val",
    )

    data_params = json_tricks.load(str(Path(datadir, "params.json")))
    os.makedirs(resultsdir, exist_ok=True)

    # Declare Loss
    def run_experiment(seed: int) -> None:
        resultspath = Path(resultsdir, f"seed-{seed}.json")
        experiment_kwargs, datapath = build_experiment_args(
            seed, datadir, kernel_name=kernel_name
        )
        experiment = Experiment(**experiment_kwargs)
        results = experiment.run()

        output = dict(
            results=results,
            datafile=str(datapath),  # Save path to datafile used
        )
        json_tricks.dump(ensure_numpy_dict(output), str(resultspath))

    Parallel(n_jobs=args.threads, verbose=10)(
        delayed(run_experiment)(i) for i in range(100)
    )
