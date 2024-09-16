import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import json_tricks
from typing import List, Dict
import os
import argparse
import numpy as np


matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "16"

# ICML one and 2 column widths
one_column_width_pts = 487.8225
one_column_width_inches = one_column_width_pts / 72.27
two_column_width_pts = 234.8775
two_column_width_inches = two_column_width_pts / 72.27


def load_results(directory: Path, num_seeds: int) -> List[Dict]:
    return [
        json_tricks.load(str(Path(directory, f"seed-{seed}.json")))
        for seed in range(num_seeds)
    ]

def plot_results(results: List[Dict], model: str, absolute: bool = False, vertical: bool = True) -> None:
    # Plot the errors as a boxplot on a single axis
    if vertical:
        fig = plt.figure(figsize=(one_column_width_inches / 3, one_column_width_inches / 2))
    else:
        fig = plt.figure(figsize=(one_column_width_inches / 3, one_column_width_inches / 2))
    # Add boxes
    if not absolute:
        boxes = [
            results_dict[model]["holdout_errors"],
            results_dict[model]["onenn_errors"],
            results_dict[model]["snn_errors"],
        ]
    else:
        boxes = [
            np.abs(results_dict[model]["holdout_errors"]),
            np.abs(results_dict[model]["onenn_errors"]),
            np.abs(results_dict[model]["snn_errors"]),
        ]
    bplot = plt.boxplot(
        boxes,
        medianprops=dict(color="k"),
        patch_artist=True,
        vert=vertical,
        widths=0.5,
    )
    color_list = ["C0", "C1", "C2"]
    for patch, c in zip(bplot["boxes"], color_list):
        try:
            patch.set_facecolor(c)
        except KeyError:
            pass

    if absolute and vertical:
        plt.ylim(0, 1.3)
        plt.xticks([], [])
        plt.ylabel("Error In Estimate (°C)")
    elif not absolute and vertical:
        plt.axhline(0, color="red", linestyle="--")
        plt.ylabel("Error In Estimate (°C)")
        plt.xticks([], [])
    elif not absolute and not vertical:
        plt.axvline(0, color="red", linestyle="--")
        plt.xlabel("Error In Estimate (°C)")
        plt.yticks([], [])
    else: 
        plt.xlim(0, 1.3)
        plt.yticks([], [])
        plt.xlabel("Error In Estimate (°C)")

    plt.tight_layout()
    if absolute and vertical:
        plt.savefig(Path(figure_directory, f"{model}vert-abs_error_boxplot.pdf"), bbox_inches="tight")
    elif not absolute and vertical:
        plt.savefig(Path(figure_directory, f"{model}vert-error_boxplot.pdf"), bbox_inches="tight")
    elif not absolute and not vertical:
        plt.savefig(Path(figure_directory, f"{model}horiz-error_boxplot.pdf"), bbox_inches="tight")
    else:
        plt.savefig(Path(figure_directory, f"{model}horiz-abs_error_boxplot.pdf"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    num_seeds = 100
    # Add argparse for grid versus metro dataset
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="metro", choices=["grid", "metro"]
    )
    parser.add_argument('--horizontal', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    results_directory = Path(
        Path(__file__).parents[1], "results", f"airTempSynthetic-{args.dataset}"
    )
    figure_directory = Path(
        Path(__file__).parents[4],
        "figures",
        "airTempSynthetic",
        f"{args.dataset}",
    )
    # Make the figure directory if it doesn't exist
    os.makedirs(figure_directory, exist_ok=True)
    # Load the results
    results = load_results(results_directory, num_seeds)
    # Extract the data
    results = [r["results"] for r in results]
    # Extract the model names
    results_dict = dict()
    for model in results[0].keys():

        model_results = [r[model] for r in results]
        sample_losses = [r["sample_loss"] for r in model_results]
        holdout_losses = [r["BasicLossEstimator_"]["estimate"] for r in model_results]
        onenn_losses = [
            r["HaversineNearestNeighborEstimator_"]["estimate"] for r in model_results
        ]
        snn_losses = [
            r["HaversineFitNearestNeighborEstimator_"]["estimate"] for r in model_results
        ]
        holdout_errors = np.array([np.abs(h - s) for h, s in zip(holdout_losses, sample_losses)])

        onenn_errors = np.array([o - s for o, s in zip(onenn_losses, sample_losses)])
        snn_errors = np.array([n - s for n, s in zip(snn_losses, sample_losses)])
        results_dict[model] = {
            "holdout_errors": holdout_errors,
            "onenn_errors": onenn_errors,
            "snn_errors": snn_errors,
        }
    for model in results_dict.keys():
        plot_results(results_dict, model, absolute=True, vertical=True)
        plot_results(results_dict, model, absolute=True, vertical=False)
        plot_results(results_dict, model, absolute=False, vertical=True)
        plot_results(results_dict, model, absolute=False, vertical=False)
