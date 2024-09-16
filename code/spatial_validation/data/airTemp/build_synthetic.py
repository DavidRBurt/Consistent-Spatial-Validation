import os
from pathlib import Path

import json_tricks
import numpy as np
from gpflow.kernels import Matern32
from gpflow.mean_functions import Constant
from gpflow.models import GPR
from gpflow.optimizers import Scipy
from spatial_validation.data.data_utils import load_dataset_from_json
from spatial_validation.models import KRR
import tensorflow as tf

if __name__ == "__main__":
    # Load data from airTempData
    path_to_data = Path(
        Path(__file__).parent, "airTempData", "metro_temperature_data.json"
    )
    path_to_grid_data = Path(
        Path(__file__).parent, "airTempData", "geo_weighted_temperature_data.json"
    )
    allwsdata = load_dataset_from_json(str(path_to_data))
    # Combine training and validation data
    all_locations = np.concatenate([allwsdata.train.S, allwsdata.validation.S], axis=0)
    all_temps = np.concatenate([allwsdata.train.Y, allwsdata.validation.Y], axis=0)

    # Interpolate with a GP with Matern 32 kernel, with params fit by max likelihood
    kern = Matern32(variance=1.0, lengthscales=np.ones(2))
    model = KRR(
        data=(all_locations, all_temps),
        kernel=kern,
        noise_variance=0.2,
    )
    opt = Scipy()
    opt.minimize(
        model.training_loss_closure(), model.trainable_variables, options={"disp": True}
    )

    num_copies = 10000
    # Add Gaussian noise to mean (note we don't sample posterior, just use mean). Use this to generate 100 responses
    test_mean_predictions = model.predict_f(allwsdata.test.S)
    train_mean_predictions = model.predict_f(allwsdata.train.S)
    val_mean_predictions = model.predict_f(allwsdata.validation.S)

    locations = np.tile(allwsdata.test.S, reps=[num_copies, 1])
    test_x = np.tile(allwsdata.test.X, reps=[num_copies, 1])
    test_mean_predictions = np.tile(test_mean_predictions, reps=[num_copies, 1])
    # Bootstrap noise distribution
    residuals = all_temps - model.predict_f(all_locations)

    path_to_save = Path(
        Path(__file__).parent, "airTempData", "synthetic", f"synthetic-metro.json"
    )
    os.makedirs(str(path_to_save.parent), exist_ok=True)
    data_dict = dict(
        train={
            "S": allwsdata.train.S,
            "X": allwsdata.train.X,
            "Y": train_mean_predictions.numpy(),
        },
        validation={
            "S": allwsdata.validation.S,
            "X": allwsdata.validation.X,
            "Y": val_mean_predictions.numpy(),
        },
        test={"S": locations, "X": test_x, "Y": test_mean_predictions},
        residuals=residuals.numpy(),
    )
    json_tricks.dump(data_dict, str(path_to_save))

    allgriddata = load_dataset_from_json(str(path_to_grid_data))

    print("Making predictions now")

    train_mean_predictions = model.predict_f(allgriddata.train.S)
    val_mean_predictions = model.predict_f(allgriddata.validation.S)
    residuals = all_temps - model.predict_f(all_locations)
    test_mean_predictions = model.predict_f(allgriddata.test.S)

    grid_path_to_save = Path(
        Path(__file__).parent, "airTempData", "synthetic", f"synthetic-grid.json"
    )
    data_dict = dict(
        train={
            "S": allgriddata.train.S,
            "X": allgriddata.train.X,
            "Y": train_mean_predictions.numpy(),
        },
        validation={
            "S": allgriddata.validation.S,
            "X": allgriddata.validation.X,
            "Y": val_mean_predictions.numpy(),
        },
        test={
            "S": allgriddata.test.S,
            "X": allgriddata.test.X,
            "Y": test_mean_predictions.numpy(),
        },
        residuals=residuals.numpy(),
    )
    json_tricks.dump(data_dict, str(grid_path_to_save))
