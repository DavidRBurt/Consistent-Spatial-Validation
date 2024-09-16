# Script for plotting the time taken by the algorithms on the synthetic data, both algorithms are plotted on a single graph for all 3 algorithmgs

import json_tricks
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse

from typing import List, Dict
from pathlib import Path

from spatial_validation.data import load_dataset_from_json

# Data, this is in the results file in dictionary format

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "7"

one_column_width_pts = 487.8225
one_column_width_inches = one_column_width_pts / 72.27
two_column_width_pts = 234.8775
two_column_width_inches = two_column_width_pts / 72.27


color_dict = {
        "BasicLossEstimator_": "C0",
        "NearestNeighborEstimator_": "C1",
        "FitNearestNeighborEstimator_": "C2",
}

def load_results(seed: int, results_directory: Path) -> Dict:
    results_path = str(Path(results_directory, f"seed-{seed}.json"))
    # extract the time take for each estimator
    results = json_tricks.load(results_path)["results"]["KRRSpatialRegression_"]
    # pop the sample loss, this isn't an estimator
    results.pop("sample_loss")
    # pop the fit nearest neighbor, something funny is happening
    # results.pop("FitNearestNeighborEstimator_")
    times = {
            est_name: results[est_name]["time_taken"]
            for est_name in results.keys()
        }
    return times


def load_dataset_results(results_directory: Path, max_seeds: int = 100) -> List:
    results = list()
    for seed in np.arange(max_seeds):
        try:
            results.append(load_results(seed, results_directory))
        except FileNotFoundError:
            print(f"Could not find seed {seed}. Plotting first {seed} seeds")
            break
    return results


if __name__ == "__main__":
    max_seeds = 100
    # add argparse
    parser = argparse.ArgumentParser()
    # add list of num_vals to argparse
    parser.add_argument(
        "-nv",
        "--num_vals",
        nargs="+",
        type=int,
        default=[250, 500, 1000, 2000, 4000, 8000],
    )
    # parse args
    args = parser.parse_args()
    ds_names = ["grid", "point"]
    model_name = "gp"
    fig_directory = Path(
        Path(__file__).parents[4], "figures", "synthetic",
    )
    for ds_name in ds_names:
        if ds_name == "grid":
            train_dist = "Cluster"
        elif ds_name == "point":
            train_dist = "Uniform"
        data_directory = Path(
            Path(__file__).parents[3],
            "spatial_validation",
            "data",
            "synthetic",
            "data",
            f"{ds_name}-{model_name}",
        )
        results_directory = Path(
            Path(__file__).parents[1], "results", f"{ds_name}-{model_name}"
        )
        results_directories = [
            Path(results_directory, f"{train_dist}-1000Train_{nv}Val")
            for nv in args.num_vals
        ]
        results = {
            nv: load_dataset_results(results_directory)
            for results_directory, nv in zip(results_directories, args.num_vals)
        }
        # Just pull out the timing data
        # Compute median times for each estimator
        median_time = {
            nv: {
                name: np.median([seed[name] for seed in results[nv]])
                for name in results[nv][0].keys()
            }
            for nv in args.num_vals
        }
        max_time = {
            nv: {
                name: np.max([seed[name] for seed in results[nv]])
                for name in results[nv][0].keys()
            }
            for nv in args.num_vals
        }
        min_time = {
            nv: {
                name: np.min([seed[name] for seed in results[nv]])
                for name in results[nv][0].keys()
            }
            for nv in args.num_vals
        }

        # Plot the median (over seeds) timing data as a function of the number of validation points,
        # Plot the max and min times as well as the median as whiskers
        fig, ax = plt.subplots()
        for name in median_time[args.num_vals[0]].keys():
            ax.errorbar(
                args.num_vals,
                [median_time[nv][name] for nv in args.num_vals],
                yerr=[
                    [median_time[nv][name] - min_time[nv][name] for nv in args.num_vals],
                    [max_time[nv][name] - median_time[nv][name] for nv in args.num_vals],
                ],
                label=name,
                color=color_dict[name],
            )

            # ax.plot(
            #     args.num_vals,
            #     [median_time[nv][name] for nv in args.num_vals],
            #     label=name,
            # )

        ax.set_xlabel("Number of Validation Points")
        ax.set_ylabel("Time Taken (s)")
        
        ax.legend(["Holdout", "1NN", "SNN (Ours)"])
        fig.tight_layout()
        fig.savefig(Path(fig_directory, f"{ds_name}-time.pdf"))
        # Plot the median (over seeds) timing data as a function of the number of validation points
        # fig, ax = plt.subplots()
        # for name in median_time[args.num_vals[0]]:
        #     ax.plot(
        #         args.num_vals,
        #         [median_time[nv][name]["knn"] for nv in args.num_vals],
        #         label=name,
        #     )



