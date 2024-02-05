import itertools
from abc import abstractmethod
from dataclasses import dataclass

import json_tricks
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.kernels import Matern32, RBF, White
from gpflow.config import default_float
from spatial_validation.rff import Matern32RFF, RBFRFF
from numpy.typing import ArrayLike
import matplotlib
import matplotlib.pyplot as plt

from spatial_validation.data.data_utils import (
    Dataset,
    SpatialDataset,
    dataset_to_dict,
    ensure_numpy_dict,
)

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "8"

# ICML one and 2 column widths
one_column_width_pts = 487.8225
one_column_width_inches = one_column_width_pts / 72.27
two_column_width_pts = 234.8775
two_column_width_inches = two_column_width_pts / 72.27


@dataclass
class SyntheticGPData:
    num_train: int
    num_val: int
    num_test: int
    spatial_dimension: int
    num_covariates: int
    covariate_lengthscale: float
    response_lengthscale: float
    noise_lengthscale: float
    spatial_noise_variance: float
    white_noise_variance: float
    kernel_name: str

    def generate_sites(self) -> ArrayLike:
        train_val_sites = self.generate_train_sites()
        test_sites = self.generate_test_sites()
        return np.concatenate([train_val_sites, test_sites])

    @abstractmethod
    def generate_train_sites(self) -> ArrayLike:
        """
        must be overwritte
        """

    @abstractmethod
    def generate_test_sites(self) -> ArrayLike:
        """
        must be overwritte
        """

    def generate_covariates(self, S: ArrayLike) -> ArrayLike:
        """
        Sample num_covariates Gaussian processes at spatial locations given by S.
        Currently these are all zero mean and have an isotropic Matern32 kernel
        """
        if self.kernel_name == "rbf":
            kernel = RBF(lengthscales=self.covariate_lengthscale) + White(
                variance=1e-12
            )  # add small amount to diagonal for stability
        else:
            kernel = Matern32(lengthscales=self.covariate_lengthscale) + White(
                variance=1e-12
            )  # add small amount to diagonal for stability
        covariance_matrix = kernel(S)
        cov_sqrt = tf.linalg.cholesky(covariance_matrix)
        dist = tfp.distributions.MultivariateNormalTriL(
            scale_tril=cov_sqrt, allow_nan_stats=False
        )
        x_transpose = dist.sample(self.num_covariates)
        return tf.transpose(x_transpose).numpy()

    def generate_response(
        self,
        S: ArrayLike,
        X: ArrayLike,
        gen_f: bool = False,
    ) -> ArrayLike:
        all_variables = tf.concat([X, S], axis=-1)
        if self.kernel_name == "rbf":
            response_kernel = RBF(
                lengthscales=self.response_lengthscale,
                active_dims=np.arange(self.num_covariates),
            )
            spatial_noise_kernel = RBF(
                lengthscales=self.noise_lengthscale,
                variance=self.spatial_noise_variance,
                active_dims=np.arange(
                    self.num_covariates, self.num_covariates + self.spatial_dimension
                ),
            )

        elif self.kernel_name == "matern32":
            response_kernel = Matern32(
                lengthscales=self.response_lengthscale,
                active_dims=np.arange(self.num_covariates),
            )
            spatial_noise_kernel = Matern32(
                lengthscales=self.noise_lengthscale,
                variance=self.spatial_noise_variance,
                active_dims=np.arange(
                    self.num_covariates, self.num_covariates + self.spatial_dimension
                ),
            )
        else:
            raise NotImplementedError
        if gen_f:
            white_noise_kernel = White(1e-6)
        else:
            white_noise_kernel = White(self.white_noise_variance)
        kernel = response_kernel + spatial_noise_kernel + white_noise_kernel

        covariance_matrix = kernel(all_variables)
        cov_sqrt = tf.linalg.cholesky(covariance_matrix)
        dist = tfp.distributions.MultivariateNormalTriL(
            scale_tril=cov_sqrt, allow_nan_stats=False
        )
        y_transpose = dist.sample(1)
        y = tf.transpose(y_transpose)
        return y.numpy()

    def generate_data(self, seed=0) -> Dataset:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        S = self.generate_sites()
        # Generate X, Sample from Multivariate Gaussian with Matern32 kernel
        X = self.generate_covariates(S=S)
        # Generate Y
        Y = self.generate_response(S, X)

        num_train_and_val = self.num_train + self.num_val

        train = SpatialDataset(
            S[: self.num_train],
            X[: self.num_train],
            Y[: self.num_train],
        )
        val = SpatialDataset(
            S[self.num_train : num_train_and_val],
            X[self.num_train : num_train_and_val],
            Y[self.num_train : num_train_and_val],
        )
        test = SpatialDataset(
            S[num_train_and_val:],
            X[num_train_and_val:],
            Y[num_train_and_val:],
        )
        data = Dataset(train, val, test)

        return data

    def save_data(self, filepath: str, seed: int = 0) -> None:
        dataset = self.generate_data(seed)
        data_dict = dataset_to_dict(dataset)
        data_dict = ensure_numpy_dict(data_dict)
        json_tricks.dump(data_dict, filepath)

    def generate_grid_sites(self, num_pts, lower, upper):
        knots_per_dim = int(
            np.floor(num_pts ** (1 / self.spatial_dimension))
        )  # number of knots per dimension
        grid_loc = np.linspace(
            lower, upper, knots_per_dim
        )  # knots along each dimension
        grid_sites = np.array(
            list(itertools.product(*(self.spatial_dimension * [grid_loc])))
        )  # tensor product
        return grid_sites

    def plot_dgp(
        self,
        num_pts,
        lower,
        upper,
        savepath,
        seed=0,
    ):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        assert self.spatial_dimension == 2
        train_val_sites = self.generate_train_sites()
        grid = self.generate_grid_sites(num_pts, lower, upper)
        x_on_grid = self.generate_covariates(grid)
        y_on_grid = self.generate_response(grid, x_on_grid, gen_f=True)
        S1 = np.sort(np.unique(grid[:, 0]))
        S2 = np.sort(np.unique(grid[:, 1]))

        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(one_column_width_inches, one_column_width_inches),
            # sharex=True,
            # sharey=True,
        )
        axes[0, 0].scatter(
            train_val_sites[: self.num_train, 0],
            train_val_sites[: self.num_train, 1],
            s=2,
            marker="o",
        )
        axes[0, 0].scatter(
            train_val_sites[self.num_train :, 0],
            train_val_sites[self.num_train :, 1],
            s=2,
            marker="x",
        )
        axes[0, 0].legend(["train", "val"], ncols=2)
        axes[0, 0].set_title("Train and Validation Sites")
        vmin, vmax = np.min(y_on_grid) - 0.1, np.max(y_on_grid) + 0.1
        x1 = np.reshape(x_on_grid[:, 0], (len(S1), len(S2)))
        axes[0, 1].imshow(
            x1,
            extent=(lower, upper, lower, upper),
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, 1].set_title("First Covariate")
        x2 = np.reshape(x_on_grid[:, 1], (len(S1), len(S2)))
        axes[1, 0].imshow(
            x2,
            extent=(lower, upper, lower, upper),
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, 0].set_title("Second Covariate")
        y2 = np.reshape(y_on_grid[:, 0], (len(S1), len(S2)))
        im = axes[1, 1].imshow(
            y2,
            extent=(lower, upper, lower, upper),
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, 1].set_title("Average Response")
        # fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
        asp = np.diff(axes[0, 0].get_xlim())[0] / np.diff(axes[0, 0].get_ylim())[0]
        axes[0, 0].set_aspect(asp)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.1, 0.04, 0.8])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig(savepath)


@dataclass
class GridTestMixin:
    s_test_min: float
    s_test_max: float

    def generate_test_sites(self):
        return self.generate_grid_sites(self.num_test, self.s_test_min, self.s_test_max)


@dataclass
class PointTestMixin:
    s_test_min: float
    s_test_max: float

    def generate_test_sites(self):
        s_test_range = self.s_test_max - self.s_test_min
        test_site_location = s_test_range * (
            np.random.rand(1, self.spatial_dimension) + self.s_test_min
        )
        test_sites = (
            np.zeros((self.num_test, self.spatial_dimension)) * test_site_location
        )
        return test_sites
