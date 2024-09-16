import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import json_tricks
from typing import List, Dict
import os
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

def plot_results(
    holdout_errors,
    onenn_errors,
    snn_errors,
    absolute: bool = False,
    vertical: bool = True,
) -> None:
    # Plot the errors as a boxplot on a single axis
    if vertical:
        fig = plt.figure(figsize=(one_column_width_inches / 3, one_column_width_inches / 2))
    else:
        fig = plt.figure(figsize=(one_column_width_inches / 2, one_column_width_inches / 2))
    # Add boxes
    if not absolute:
        boxes = [
            holdout_errors,
            onenn_errors,
            snn_errors,
        ]
    else:
        boxes = [
            np.abs(holdout_errors),
            np.abs(onenn_errors),
            np.abs(snn_errors),
        ]
    bplot = plt.boxplot(
        boxes,
        medianprops=dict(color="k"),
        patch_artist=True,
        vert=vertical,
        widths=0.4,
    )
    color_list = ["C0", "C1", "C2"]
    for patch, c in zip(bplot["boxes"], color_list):
        try:
            patch.set_facecolor(c)
        except KeyError:
            pass
    # Set limits and labels
    if vertical:
        plt.ylabel("Error In Estimate (m/s)")
        plt.xticks([], [])
        if absolute:
            plt.ylabel("Error In Estimate (m$^2$/s$^2$)")
            plt.ylim(0, 3.2)
        else:
            plt.ylabel("Error In Estimate (m/s)")
            plt.axhline(0, color="red", linestyle="--")
    else:
        if absolute:
            plt.xlabel("Error In Estimate ($m^2/s^2$)")
        else:
            plt.xlabel("Error In Estimate (m/s)")
        plt.yticks([], [])
        if absolute:
            plt.xlim(0, 3.2)
        else:
            plt.axvline(0, color="red", linestyle="--")
    plt.tight_layout()

    # save figure with name depending on options
    if absolute and vertical:
        plt.savefig(Path(figure_directory, "abs_error_boxplot.pdf"), bbox_inches='tight')
    elif vertical:
        plt.savefig(Path(figure_directory, "error_boxplot.pdf"), bbox_inches='tight')
    elif absolute:
        plt.savefig(Path(figure_directory, "abs_error_boxplot_horizontal.pdf"), bbox_inches='tight')
    else:
        plt.savefig(Path(figure_directory, "error_boxplot_horizontal.pdf"), bbox_inches='tight')
    plt.close()




if __name__ == "__main__":
    num_seeds = 100
    results_directory = Path(Path(__file__).parents[1], "results")
    figure_directory = Path(Path(__file__).parents[4], "figures", "windSpeed")
    # Make the figure directory if it doesn't exist
    os.makedirs(figure_directory, exist_ok=True)
    # Load the results
    results = load_results(results_directory, num_seeds)
    # Extract the data
    results = [r["results"] for r in results]
    results = [r["GBMSpatialModel_"] for r in results]
    sample_losses = [0.1 * np.sqrt(r["sample_loss"]) for r in results]
    holdout_losses = [0.1 * np.sqrt(r["BasicLossEstimator_"]["estimate"]) for r in results]
    onenn_losses = [0.1 * np.sqrt(r["HaversineNearestNeighborEstimator_"]["estimate"]) for r in results]
    snn_losses = [0.1 * np.sqrt(r["HaversineFitNearestNeighborEstimator_"]["estimate"]) for r in results]

    sample_losses_rmse = np.sqrt(sample_losses)
    holdout_losses_rmse = np.sqrt(holdout_losses)
    onenn_losses_rmse = np.sqrt(onenn_losses)
    snn_losses_rmse = np.sqrt(snn_losses)
    
    holdout_mse_errors = np.array([h - s for h, s in zip(holdout_losses, sample_losses)])
    onenn_mse_errors = np.array([o - s for o, s in zip(onenn_losses, sample_losses)])
    snn_mse_errors = np.array([n - s for n, s in zip(snn_losses, sample_losses)])

    holdout_rmse_errors = np.array([h - s for h, s in zip(holdout_losses_rmse, sample_losses_rmse)])
    onenn_rmse_errors = np.array([o - s for o, s in zip(onenn_losses_rmse, sample_losses_rmse)])
    snn_rmse_errors = np.array([n - s for n, s in zip(snn_losses_rmse, sample_losses_rmse)])

    plot_results(holdout_mse_errors, onenn_mse_errors, snn_mse_errors, absolute=True, vertical=True)
    plot_results(holdout_rmse_errors, onenn_rmse_errors, snn_rmse_errors, absolute=False, vertical=True)

