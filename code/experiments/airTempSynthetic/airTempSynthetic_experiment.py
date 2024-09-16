import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import json_tricks
from joblib import Parallel, delayed
import numpy as np
from numpy.typing import ArrayLike
from gpflow.kernels import Matern32

from spatial_validation.data import (
    ensure_numpy_dict,
    load_dataset_from_json,
    Dataset,
    SpatialDataset,
)

from spatial_validation.experiment import Experiment
from spatial_validation.losses import TruncatedAbsoluteError
from spatial_validation.models import GeographicallyWeightedRegression


def build_kernel(spatial_scales: ArrayLike, kernel_scale: float):
    return Matern32(
        lengthscales=spatial_scales, variance=kernel_scale**2, active_dims=(2, 3)
    )


if __name__ == "__main__":
    # Load data from json
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="grid",
        choices=[
            "metro",
            "grid",
        ],
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    datadir = Path(
        Path(__file__).parents[2],
        "spatial_validation",
        "data",
        "airTemp",
        "airTempData",
        "synthetic",
        f"synthetic-{args.dataset}.json",
    )

    resultsdir = Path(
        Path(__file__).parent, "results", f"airTempSynthetic-{args.dataset}"
    )
    os.makedirs(resultsdir, exist_ok=True)
    # Declare Loss
    loss = TruncatedAbsoluteError(5, scale_factor=6371 / 100)
    # Load the data
    dataset = load_dataset_from_json(str(datadir), bootstrap=True)
    # Estimators to Use
    estimator_names = [
        "BasicLossEstimator",
        "HaversineNearestNeighborEstimator",
        "HaversineFitNearestNeighborEstimator",
    ]
    estimator_parameters = [
        dict(),
        dict(),
        dict(),
    ]
    # Models and parameters
    model_names = [
        "GeographicallyWeightedRegression",
        "KRRSpatialRegression",
    ]

    param_names = ["", ""]

    def run_experiment(seed: int) -> None:
        ds = dataset(seed)
        kernel_scale = ds.train.Y.std()
        likelihood_scale = kernel_scale * np.sqrt(0.1)
        spatial_scales = ds.train.S.std(axis=0)
        model_parameters = [
            dict(),
            dict(
                response_kernel=build_kernel(spatial_scales, kernel_scale),
                regularization_parameter=likelihood_scale**2,
                center=True,
                max_likelihood=True,
            ),
        ]
        experiment = Experiment(
            dataset=ds,
            model_names=model_names,
            model_parameters=model_parameters,
            param_names=param_names,
            estimator_names=estimator_names,
            estimator_parameters=estimator_parameters,
            loss=loss,
        )

        results = experiment.run()
        output = dict(
            results=results,
            datafile=str(datadir),  # Save path to datafile used
        )
        results_path = str(Path(resultsdir, f"seed-{seed}.json"))
        json_tricks.dump(ensure_numpy_dict(output), results_path)

    Parallel(n_jobs=args.threads, verbose=10)(
        delayed(run_experiment)(i) for i in range(100)
    )