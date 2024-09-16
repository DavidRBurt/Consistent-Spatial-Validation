import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import json_tricks
from joblib import Parallel, delayed
import numpy as np
from numpy.typing import ArrayLike

from spatial_validation.data.windSpeed.winddataset import WindDataset
from spatial_validation.data import Dataset, SpatialDataset, ensure_numpy_dict

from spatial_validation.experiment import Experiment
from spatial_validation.losses import AbsoluteError, SquaredLoss, TruncatedSquaredLoss
from spatial_validation.models import GBMSpatialModel


def to_radians(ds) -> Dataset:
    def _to_rads(sd) -> SpatialDataset:
        return SpatialDataset(np.radians(sd.S), sd.X, sd.Y)

    return Dataset(_to_rads(ds.train), _to_rads(ds.validation), _to_rads(ds.test))


if __name__ == "__main__":

    datadir = Path(
        Path(__file__).parents[2],
        "spatial_validation",
        "data",
        "windSpeed",
        "data",
        "wind_speed_data.csv",
    )

    resultsdir = Path(
        Path(__file__).parent,
        "results",
    )
    os.makedirs(resultsdir, exist_ok=True)
    # Declare Loss, remember units are .1m/s, this amounts to 0.1*sqrt(2500) = 5m/s
    loss = TruncatedSquaredLoss(2500, scale_factor=6371)
    # Load the data
    ds = WindDataset()
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
        "GBMSpatialModel",
    ]
    param_names = [""]

    def run_experiment(seed: int = 0) -> None:
        dataset = ds(seed)
        dataset = to_radians(dataset)
        # Compute scale factors to initialize GP (note Ytrain mean is small, no need to center for GP fitting)

        model_parameters = [
            dict(
                num_leaves=127,
                n_estimators=100,
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
            oracle = np.var(dataset.test.Y), # best we could do, check model isn't so bad
            datafile=str(datadir),  # Save path to datafile used
        )
        results_path = str(Path(resultsdir, f"seed-{seed}.json"))
        json_tricks.dump(ensure_numpy_dict(output), results_path)

    Parallel(n_jobs=15, verbose=10)(
        delayed(run_experiment)(i) for i in range(100)
    )
