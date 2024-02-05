from spatial_validation.losses import TruncatedSquaredLoss
from spatial_validation.estimators import NearestNeighborEstimator, BasicLossEstimator
from spatial_validation.models import ConstantPredictor, MAELinearModel
import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import json_tricks
import numpy as np

from spatial_validation.data import ensure_numpy_dict, load_dataset_from_json
from spatial_validation.experiment import Experiment
from spatial_validation.losses import AbsoluteError


def build_experiment_args(seed: int, datadir: Path) -> Dict:
    datapath = Path(datadir, f"seed-{seed}.json")
    dataset = load_dataset_from_json(str(datapath))

    loss = AbsoluteError()
    # Estimators to Use
    estimator_names = [
        "BasicLossEstimator",
        "NearestNeighborEstimator",
        "FitNearestNeighborEstimator",
    ]
    estimator_parameters = [dict(), dict(), dict()]
    # Models and parameters

    model_names = ["ConstantPredictor", "MAELinearModel"]
    model_parameters = [dict(value=0.25), dict()]
    return (
        dict(
            dataset=dataset,
            model_names=model_names,
            model_parameters=model_parameters,
            param_names=["", ""],
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
        "-nv",
        "--num_val",
        type=int,
        default=500,
    )
    # add argparse to choose which seeds to run
    parser.add_argument("-t", "--threads", type=int, default=10)
    # parse args
    args = parser.parse_args()
    # Load Data
    datadir = Path(
        Path(__file__).parents[2],
        "spatial_validation",
        "data",
        "model_selection",
        "data",
        f"nval_{args.num_val}",
    )
    resultsdir = Path(
        Path(__file__).parent,
        "results",
        "model_selection",
        f"nval_{args.num_val}",
    )
    os.makedirs(resultsdir, exist_ok=True)

    def run_experiment(seed: int) -> None:
        resultspath = Path(resultsdir, f"seed-{seed}.json")
        experiment_kwargs, datapath = build_experiment_args(seed, datadir)
        experiment = Experiment(**experiment_kwargs)
        results = experiment.run()
        output = dict(
            results=results,
            datafile=str(datapath),  # Save path to datafile used
        )
        json_tricks.dump(ensure_numpy_dict(output), str(resultspath))

    # Fast enough no need to run in parallel
    for i in range(100):
        run_experiment(i)
