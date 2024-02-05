from pathlib import Path
from typing import Dict, List
import argparse
import os

import json_tricks
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from spatial_validation.data import Dataset
from spatial_validation.data import load_dataset_from_json
from spatial_validation.models import MAELinearModel

# Set params
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "9"

# ICML one and 2 column widths
one_column_width_pts = 487.8225
one_column_width_inches = one_column_width_pts / 72.27
two_column_width_pts = 234.8775
two_column_width_inches = two_column_width_pts / 72.27


def load_results(seed: int, results_directory: Path) -> Dict:
    results_path = str(Path(results_directory, f"seed-{seed}.json"))
    return json_tricks.load(results_path)


def load_all_results(results_directory: Path):
    return [load_results(seed, results_directory)["results"] for seed in range(100)]


def selected_constant_model(all_results: List[Dict]) -> Dict:
    holdout_estimates_constant = [
        r["ConstantPredictor_"]["BasicLossEstimator_"]["estimate"] for r in all_results
    ]
    holdout_estimates_linear = [
        r["MAELinearModel_"]["BasicLossEstimator_"]["estimate"] for r in all_results
    ]
    holdout_selects_constant = [
        c < l for c, l in zip(holdout_estimates_constant, holdout_estimates_linear)
    ]
    onenn_estimates_constant = [
        r["ConstantPredictor_"]["NearestNeighborEstimator_"]["estimate"] for r in all_results
    ]
    onenn_estimates_linear = [
        r["MAELinearModel_"]["NearestNeighborEstimator_"]["estimate"] for r in all_results
    ]
    onenn_selects_constant = [
        c < l for c, l in zip(onenn_estimates_constant, onenn_estimates_linear)
    ]
    fitnn_estimates_constant = [
        r["ConstantPredictor_"]["FitNearestNeighborEstimator_"]["estimate"] for r in all_results
    ]
    fitnn_estimates_linear = [
        r["MAELinearModel_"]["FitNearestNeighborEstimator_"]["estimate"] for r in all_results
    ]
    fitnn_selects_constant = [
        c < l for c, l in zip(fitnn_estimates_constant, fitnn_estimates_linear)
    ]
    sample_loss_constant = [r["ConstantPredictor_"]["sample_loss"] for r in all_results]
    sample_loss_linear = [r["MAELinearModel_"]["sample_loss"] for r in all_results]
    sample_loss_selects_constant = [
        c < l for c, l in zip(sample_loss_constant, sample_loss_linear)
    ]
    return dict(
        holdout=holdout_selects_constant,
        onenn=onenn_selects_constant,
        fitnn=fitnn_selects_constant,
        sample_loss=sample_loss_selects_constant,
    )


if __name__ == "__main__":
    results_directory = Path(Path(__file__).parents[1], "results", "model_selection")
    figdir = Path(Path(__file__).parents[4], "figures", "model_selection")
    os.makedirs(figdir, exist_ok=True)
    holdout_frac = list()
    onenn_frac = list()
    fitnn_frac = list()
    nvals = [5, 15, 25, 35, 45, 55, 65, 75]
    for n_val in nvals:
        full_results_directory = Path(results_directory, f"nval_{n_val}")
        all_results = load_all_results(full_results_directory)
        model_selected = selected_constant_model(all_results)
        # Check constant is best
        assert all(t for t in model_selected["sample_loss"])
        # count number of times constant is selected
        holdout_selections = np.sum(model_selected["holdout"])
        onenn_selections = np.sum(model_selected["onenn"])
        fitnn_selections = np.sum(model_selected["fitnn"])
        holdout_frac.append(holdout_selections)
        onenn_frac.append(onenn_selections)
        fitnn_frac.append(fitnn_selections)

    plt.figure(figsize=(two_column_width_inches, two_column_width_inches / 2))
    plt.plot(nvals, holdout_frac, "C0")
    plt.plot(nvals, onenn_frac, "C1")
    plt.plot(nvals, fitnn_frac, "C2")
    plt.legend(["Holdout", "1NN", "SNN"], ncols=1
    )
    plt.ylabel("$\%$ seeds $h_0$ Selected")
    plt.xlabel("Number of Points used for Validation")
    plt.tight_layout()
    figpath = str(Path(figdir, "model_selection_percent.pdf"))
    plt.savefig(figpath)
    plt.close()

    n_val = 15
    seed = 0
    full_results_directory = Path(results_directory, f"nval_{n_val}")
    results = load_results(seed, full_results_directory)
    datapath = results["datafile"]
    data = load_dataset_from_json(datapath)
    plt.figure(figsize=(two_column_width_inches, two_column_width_inches / 2))
    plt.scatter(data.train.S, data.train.Y, s=5)
    plt.scatter(data.validation.S, data.validation.Y, s=5)
    plt.scatter(data.test.S, data.test.Y, s=5)
    plt.hlines(y=0.25, xmin=0, xmax=1, colors="r", ls="--")
    plt.xlim([0, 1])
    m = MAELinearModel(data.train)
    preds = m.predict(data.test.S, None)
    plt.plot(data.test.S, preds, c="k", ls="--")
    plt.legend(["Train", "Val", "Test", "$h_0$", "$h_1$"], ncols=3, loc="lower left", fontsize=8, borderpad=0.25)
    figpath = str(Path(figdir, "model_selection_data.pdf"))
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()
