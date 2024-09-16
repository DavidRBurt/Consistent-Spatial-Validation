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
from spatial_validation.data.UKHousePrices.UKHousingPricesDataset import UKHousingDataset
from spatial_validation.experiment import Experiment
from spatial_validation.losses import AbsoluteExpError, TruncatedAbsoluteExpError
from spatial_validation.models import LowRankKRRSpatialRegression


def build_kernel(spatial_scales: ArrayLike, kernel_scale: float):
    return Matern32(
        lengthscales=spatial_scales / 2.0, variance=kernel_scale**2, active_dims=(0, 1)
    ) + Matern32(
        lengthscales=spatial_scales * 2.0, variance=kernel_scale**2, active_dims=(0, 1)
    )


def to_radians(ds) -> Dataset:
    def _to_rads(sd) -> SpatialDataset:
        return SpatialDataset(np.radians(sd.S), sd.X, sd.Y)

    return Dataset(_to_rads(ds.train), _to_rads(ds.validation), _to_rads(ds.test))


if __name__ == "__main__":
    # Load data from json
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    datadir = Path(
        Path(__file__).parents[2],
        "spatial_validation",
        "data",
        "UKHousePrices",
        "data",
        "UKHousingData.json",
    )

    resultsdir = Path(
        Path(__file__).parent,
        "results-truncated",
    )
    os.makedirs(resultsdir, exist_ok=True)
    # Declare Loss
    # rescale max loss by 1000 to adjust for implausible Lipschitz constant
    loss = TruncatedAbsoluteExpError(1e6, scale_factor=float(1e3) * 6371) 
    # Load the data
    ds = UKHousingDataset(ntest=1000, ntrain=40000)
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
        "LowRankKRRSpatialRegression",
    ]
    param_names = [""]


    def run_experiment(seed: int) -> None:
        dataset = ds(seed)
        dataset = to_radians(dataset)
        # Compute scale factors to initialize GP (note Ytrain mean is small, no need to center for GP fitting)
        kernel_scale = dataset.train.Y.std()
        likelihood_scale = kernel_scale * np.sqrt(0.1)
        spatial_scales = dataset.train.S.std(axis=0)


        model_parameters = [
            dict(
                response_kernel=build_kernel(spatial_scales, kernel_scale),
                regularization_parameter=likelihood_scale**2,
                center=True,
                max_likelihood=True,
                num_inducing=2000,
            ),
        ]


        experiment = Experiment(
            dataset=dataset,
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
    
    Parallel(n_jobs=5, verbose=10)(
        delayed(run_experiment)(i) for i in range(100)
    )
