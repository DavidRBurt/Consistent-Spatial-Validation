import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from joblib import Parallel, delayed

import json_tricks
import matplotlib.pyplot as plt
import numpy as np
from gp_data_generator import (
    GridTestMixin,
    SyntheticGPData,
    PointTestMixin,
)


@dataclass
class ClusteredProcessMixin:
    def generate_train_sites(self):
        mixture_centers = list()
        mixture_stds = list()
        mixture_weights = list()
        sites = list()

        def add_cluster():
            mixture_centers.append(
                np.random.uniform(-0.5, 0.5, (self.spatial_dimension,))
            )
            mixture_stds.append(np.random.uniform(0.05, 0.15))
            mixture_weights.append(1.0)

        def sample_cluster(cluster_choice: int):
            std = mixture_stds[cluster_choice]
            mean = mixture_centers[cluster_choice]
            new_point = (
                np.random.randn(
                    self.spatial_dimension,
                )
                * std
                + mean
            )
            sites.append(new_point)
            mixture_weights[cluster_choice] += 1.0 / mixture_weights[cluster_choice]

        add_cluster()
        sample_cluster(0)

        for i in np.arange(self.num_train + self.num_val - 1):
            weights = np.concatenate([mixture_weights, [1.0]])
            cluster_choice = np.random.choice(
                len(mixture_centers) + 1, p=weights / np.sum(weights)
            )
            if cluster_choice == len(mixture_centers):
                add_cluster()
            sample_cluster(cluster_choice)
        # stack sites into array
        sites = np.stack(sites, axis=0)
        return sites


@dataclass
class UniformMixin:
    def generate_train_sites(self):
        total = self.num_train + self.num_val
        return np.random.rand(total, self.spatial_dimension) - 0.5


@dataclass
class SyntheticClusterTrainGridTest(
    ClusteredProcessMixin, GridTestMixin, SyntheticGPData
):
    pass


@dataclass
class SyntheticClusterTrainPointTest(UniformMixin, PointTestMixin, SyntheticGPData):
    pass


if __name__ == "__main__":
    # add argparse to decide which type of dataset to generate
    datadir = Path(__file__).parent

    parser = argparse.ArgumentParser()
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
    parser.add_argument("-k", "--kernel", type=str, default="matern32", choices=["rbf", "matern32", "matern12"])
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction)

    parser.add_argument(
        "-nv",
        "--num_val",
        type=int,
        default=500,
    )
    parser.add_argument(
        "-nt",
        "--num_train",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=5,
    )

    # parse arguments
    args = parser.parse_args()
    
    model_name = "gp"
    kernel_name = args.kernel

    if args.dataset == "cluster":
        data_args = dict(
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=30**2,
            num_test_copies=50,
            spatial_dimension=2,
            num_covariates=2,
            covariate_lengthscale=0.3,
            response_lengthscale=1.0,
            noise_lengthscale=0.5,
            spatial_noise_variance=0.5,
            white_noise_variance=0.1,
            s_test_min=-0.5,
            s_test_max=0.5,
            kernel_name=kernel_name,
        )
        ds_name = "grid"
        train_dist = "Cluster"
        data_generator = SyntheticClusterTrainGridTest(**data_args)
    elif args.dataset == "point_prediction":
        data_args = dict(
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=1,
            num_test_copies=45000,
            spatial_dimension=2,
            num_covariates=2,
            covariate_lengthscale=0.3,
            response_lengthscale=1.0,
            noise_lengthscale=0.5,
            spatial_noise_variance=0.5,
            white_noise_variance=0.1,
            s_test_min=-0.5,
            s_test_max=0.5,
            kernel_name=kernel_name,
        )
        ds_name = "point"
        train_dist = "Uniform"
        data_generator = SyntheticClusterTrainPointTest(**data_args)

    savedir = Path(
            datadir,
            "data",
            f"{ds_name}-{model_name}",
            f"{train_dist}-{args.num_train}Train_{args.num_val}Val",
    )
    os.makedirs(str(savedir), exist_ok=True)
    # save parameters in file for posterity
    param_fp = str(Path(savedir, "params.json"))
    json_tricks.dump(data_args, param_fp)

    # Save 100 seeds
    def generate_data(seed: int):
        data_fp = str(Path(savedir, f"seed-{seed}.json"))
        data_generator.save_data(filepath=data_fp, seed=seed)

    ds = data_generator.generate_data(0)
    if args.plot:
        figdir = Path(Path(__file__).parents[4], "figures", "synthetic", f"{ds_name}-{model_name}")
        os.makedirs(str(figdir), exist_ok=True)
        for seed in range(3):
            fig_path = Path(figdir, f"data-{seed}.pdf")
            data_generator.plot_dgp(2500, lower=-0.5, upper=0.5, savepath=str(fig_path), seed=seed)
    else:
        Parallel(n_jobs=args.threads, verbose=10)(
            delayed(generate_data)(i) for i in range(100)
        )
