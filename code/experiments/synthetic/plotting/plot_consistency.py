import argparse
import os
from pathlib import Path
from typing import Dict, List

import json_tricks
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from spatial_validation.data import load_dataset_from_json

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "7"

# ICML one and 2 column widths
one_column_width_pts = 487.8225
one_column_width_inches = one_column_width_pts / 72.27
two_column_width_pts = 234.8775
two_column_width_inches = two_column_width_pts / 72.27


def load_results(seed: int, results_directory: Path) -> Dict:
    results_path = str(Path(results_directory, f"seed-{seed}.json"))
    return json_tricks.load(results_path)


def plot_risks(
    all_results: Dict[int, List[Dict]],
    fig_directory: Path,
    model_name: str,
    upper: float = 0.5,
):
    all_abs_boxes = list()
    all_rel_boxes = list()
    box_positions = list()
    box_labels = list()
    color_dict = {
        "BasicLossEstimator_": "C0",
        "NearestNeighborEstimator_": "C1",
        "FitNearestNeighborEstimator_": "C2",
    }
    for i, (nv, results) in enumerate(all_results.items()):
        relative_signed_diffs = dict()
        absolute_diffs = dict()
        for seed in np.arange(len(results)):
            cur_results = results[seed]["results"][model_name]
            sample_loss = cur_results["sample_loss"]
            for estimator_name in cur_results.keys():
                # Get the estimated risk
                if estimator_name == "sample_loss":
                    continue
                if estimator_name not in absolute_diffs.keys():
                    absolute_diffs[estimator_name] = list()
                    relative_signed_diffs[estimator_name] = list()
                if (
                    isinstance((cur_results[estimator_name]["estimate"] - sample_loss), np.ndarray)
                    and (cur_results[estimator_name]["estimate"] - sample_loss).ndim > 0
                ):
                    assert len((cur_results[estimator_name]["estimate"] - sample_loss)) == 1
                relative_signed_diffs[estimator_name].append(
                    np.squeeze(cur_results[estimator_name]["estimate"] - sample_loss) / sample_loss
                )
                absolute_diffs[estimator_name].append(
                    np.squeeze(np.abs(cur_results[estimator_name]["estimate"] - sample_loss))
                )
        offset_increment = 0.5
        offset = -0.5
        for estimator_name in absolute_diffs.keys():
            all_abs_boxes.append(absolute_diffs[estimator_name])
            all_rel_boxes.append(relative_signed_diffs[estimator_name])
            # print(nv, 2 * i + offset, estimator_name)
            box_positions.append(2 * i + offset)
            box_labels.append(estimator_name)
            offset += offset_increment

        # make figure and get axes
    fig = plt.figure(figsize=(one_column_width_inches, one_column_width_inches / 2))
    ax = fig.get_axes()
    plt.xlabel("Number of Validation Points")
    plt.ylabel(
        "$(\hat{R}_{Q^{\mathrm{test}}}(h) - \hat{R}(h))/\hat{R}^{X}_{Q^{\mathrm{test}}}(h)$"
    )
    bplot = plt.boxplot(
        all_rel_boxes,
        medianprops=dict(color="k"),
        patch_artist=True,
        positions=box_positions,
        widths=0.3,
    )
    for patch, label in zip(bplot["boxes"], box_labels):
        try:
            patch.set_facecolor(color_dict[label])
        except KeyError:
            pass

    # add red line at 0 error covering extent of x range
    plt.hlines(
        0, -1.0, 2 * len(all_results.keys()) - 1, linestyle="dashed", colors="C3"
    )
    plt.xticks(ticks=2 * np.arange(len(all_results.keys())), labels=all_results.keys())
    plt.xlim(-1.0, 2 * len(all_results.keys()) - 1)
    # add legend for two estimators
    # plt.legend(
    #     handles=[bplot["boxes"][0], bplot["boxes"][1], bplot["boxes"][2]],
    #     labels=["Holdout", "1NN", "SNN"],
    #     ncol=3,
    # )

    plt.tight_layout()
    plt.savefig(str(Path(fig_directory, "rel_risk_boxplot.pdf")))
    plt.close()

    fig = plt.figure(figsize=(two_column_width_inches, two_column_width_inches / 1.75))
    ax = fig.get_axes()
    plt.xlabel("Number of Validation Points")
    plt.ylabel("$|\hat{R}_{Q^{\mathrm{test}}}(h) - \hat{R}(h)|$")
    bplot = plt.boxplot(
        all_abs_boxes,
        medianprops=dict(color="k"),
        patch_artist=True,
        positions=box_positions,
        widths=0.3,
    )

    for patch, label in zip(bplot["boxes"], box_labels):
        try:
            patch.set_facecolor(color_dict[label])
        except KeyError:
            pass

    plt.xticks(ticks=2 * np.arange(len(all_results.keys())), labels=all_results.keys())
    plt.xlim(-1.0, 2 * len(all_results.keys()) - 1)
    plt.gca().set_ylim(bottom=0, top=upper)
    for i, (b, p) in enumerate(zip(all_abs_boxes, box_positions)):
        num_out = np.sum(np.array(b) > upper)
        if num_out > 0:
            plt.text(
                p - 0.25,
                upper + 0.01,
                f"+{num_out}",
                fontdict={"color": f"C{i % 3}", "size": 5},
            )
    # add legend for two estimators
    # plt.legend(
    #     handles=[bplot["boxes"][0], bplot["boxes"][1], bplot["boxes"][2]],
    #     labels=["Holdout", "1NN", "SNN"],
    #     ncol=3,
    # )

    plt.tight_layout()
    plt.savefig(str(Path(fig_directory, "abs_risk_boxplot.pdf")))
    plt.close()


def plot_site_distribution(seed: int, datasets, fig_directory: Path):
    # plot the site distribution for the first seed
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(two_column_width_inches, two_column_width_inches / 1.3),
        sharex=True,
        sharey=True,
    )
    flat_axes = np.reshape(axes, -1)
    for i, (dataset, ax) in enumerate(zip(datasets, flat_axes)):
        ax.scatter(
            dataset.validation.S[:, 0],
            dataset.validation.S[:, 1],
            c="C0",
            marker="o",
            s=0.25,
            alpha=0.3,
            label="Validation",
            rasterized=True,
        )
        # remove ticks from inner plots
        if i == 0:
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=True,
                right=False,
            )
        if i in [1, 2]:
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
            )
        if i in [4, 5]:
            ax.tick_params(
                axis="both",
                which="both",
                bottom=True,
                top=False,
                left=False,
                right=False,
            )
        # Set axis labels on outer plots
        if i in [3, 4, 5]:
            ax.set_xlabel("S1")
        if i in [0, 3]:
            ax.set_ylabel("S2")
        # set size of markers to be 1, rasterize to make pdf smaller
        ax.scatter(
            dataset.test.S[:, 0],
            dataset.test.S[:, 1],
            c="C1",
            s=0.25,
            marker="o",
            alpha=0.3,
            label="Test",
            rasterized=True,
        )
        # set axis limits
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        # put legend above top right plots
        if i == 1:
            # ax.legend(markerscale=5, loc="upper right", fontsize=8, ncols=2)
            # Move legend above plot
            ax.legend(
                markerscale=5,
                loc="upper center",
                fontsize=8,
                ncol=2,
                bbox_to_anchor=(0.5, 1.3),
            )
    # Reduce gaps between plots
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    # Reduce gap between plots and title
    plt.subplots_adjust(top=0.85)

    plt.savefig(str(Path(fig_directory, f"site-distribution-seed-{seed}.pdf")), dpi=500)


def plot_kstar(all_results: Dict[int, List[Dict]], fig_directory: Path) -> None:
    nvals = all_results.keys()
    def extract_kstars(nval):
        result_list = all_results[nval]
        return [result["results"]['KRRSpatialRegression_']["FitNearestNeighborEstimator_"]["n_neighbors"] for result in result_list]
    
    kstars = [extract_kstars(nval) for nval in nvals]
    col_names = nvals
    row_names = np.unique(np.array(kstars))
    table = np.zeros((len(row_names), len(col_names)))
    for j, kstar in enumerate(kstars):
        unique, counts = np.unique(kstar, return_counts=True)
        for ell, k in enumerate(row_names):
            for i, u in enumerate(unique):
                if k == u:
                    table[ell, j] = counts[i]


    strs = list()
    strs.append("\\begin{tabular}{c cccccc} \n")
    strs.append("  \multicolumn{6}{c@{}}{Number of Validation Points} \\\\ \cmidrule(l){2-7}")
    strs.append("  $\\mathbf{\kstar}$  & "+ " & ".join([str(c) for c in col_names]) + " \\\\ \hline \n")
    for i, r in enumerate(row_names):
        strs.append(f"{r} &" +  " & ".join([str(int(t)) for t in table[i]]) + " \\\\ \n")

    strs.append("\\end{tabular}")
    tablefile = Path(fig_directory, "k_chosen.tex")
    with open(tablefile, "w") as f:
        f.writelines(strs)
    # plt.boxplot(kstars)
    # plt.savefig("tmp.png")

if __name__ == "__main__":
    max_seeds = 100
    # add argparse
    parser = argparse.ArgumentParser()
    # add dataset to argparse
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cluster",
        choices=[
            "cluster",
            "point_prediction",
        ],
    )
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

    if args.dataset == "cluster":
        ds_name = "grid"
        train_dist = "Cluster"
    elif args.dataset == "point_prediction":
        ds_name = "point"
        train_dist = "Uniform"
    else:
        raise ValueError("Invalid dataset choice")
    model_name = "gp"

    fig_directory = Path(
        Path(__file__).parents[4], "figures", "synthetic", f"{ds_name}-{model_name}"
    )
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
    data_directories = [
        Path(data_directory, f"{train_dist}-1000Train_{nv}Val") for nv in args.num_vals
    ]
    results_directories = [
        Path(results_directory, f"{train_dist}-1000Train_{nv}Val")
        for nv in args.num_vals
    ]
    # set up directories
    os.makedirs(fig_directory, exist_ok=True)

    def load_dataset_results(results_directory: Path):
        results = list()
        for seed in np.arange(max_seeds):
            try:
                results.append(load_results(seed, results_directory))
            except FileNotFoundError:
                print(f"Could not find seed {seed}. Plotting first {seed} seeds")
                break
        return results

    all_results = {
        nv: load_dataset_results(results_directory)
        for results_directory, nv in zip(results_directories, args.num_vals)
    }
    mn = "KRRSpatialRegression_"
    upper = 0.5 if (args.dataset == "point_prediction") else 0.15
    plot_risks(all_results, fig_directory, model_name=mn, upper=upper)
    # Plot the site distributions for the first seed
    seed = 1
    datasets = [
        load_dataset_from_json(str(Path(data_directory, f"seed-{seed}.json")))
        for data_directory in data_directories
    ]
    plot_kstar(all_results, fig_directory)
    plot_site_distribution(0, datasets, fig_directory)
    plot_site_distribution(1, datasets, fig_directory)
