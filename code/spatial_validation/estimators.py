from abc import abstractmethod
from typing import Callable, Optional, Union, Tuple, Dict

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.kernels import Kernel, Matern32
from gpflow.utilities import add_noise_cov
from numpy.typing import ArrayLike
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity, KNeighborsRegressor, NearestNeighbors

from .data import SpatialDataset
from .losses import Loss, SquaredLoss, TruncatedSquaredLoss
from .models import SpatialModel
from .compute_fill_distance import approximate_fill_distance


def _to_np(arr: Union[ArrayLike, tf.Tensor]):
    """
    Converts an input array to a numpy array if it is not already one.
    """
    return arr if isinstance(arr, np.ndarray) else arr.numpy()


class LossEstimator:
    """
    A class for estimating the loss of a spatial model on validation data.

    Attributes:
    -----------
    validation_data : SpatialDataset
        The validation dataset to use for estimating the loss.
    test_sites : ArrayLike
        The test sites we want to estimate the loss at.
    loss : Callable
        The loss function.
    """

    def __init__(
        self, validation_data: SpatialDataset, test_sites: ArrayLike, loss: Callable
    ):
        self.validation_data = validation_data
        self.test_sites = test_sites
        self.loss = loss

    @abstractmethod
    def estimate_risk(
        self, model: SpatialModel, weights: Optional[ArrayLike] = None
    ) -> float:
        """
        Estimates the risk associated with the given spatial model.

        Args:
            model (SpatialModel): The spatial model to estimate risk for.

        Returns:
            float: The estimated risk associated with the given spatial model.
        """
        pass

    @abstractmethod
    def estimate_confidence(
        self, model: SpatialModel, use_cached: bool = True
    ) -> float:
        """
        Estimates the standard error of the estimate, only defined for BasicLossEstimator.

        Args:
            model (SpatialModel): The spatial model to estimate confidence for.

        Returns:
            float: The estimated std error of the estimate associated with the given spatial model.
        """
        return None


class WeightedEstimator(LossEstimator):
    def __init__(
        self,
        validation_data: SpatialDataset,
        test_sites: ArrayLike,
        loss: Callable,
        return_confidence: bool = False,
        **kwargs,
    ):
        super().__init__(validation_data, test_sites, loss)
        self.weights = self.build_weights(**kwargs)

    @abstractmethod
    def build_weights(self) -> ArrayLike:
        """
        Return vector of weights same shape as validation sites

        Returns:
            ArrayLike: [description]
        """
        pass

    def estimate_risk(
        self, model: SpatialModel, weights: Optional[ArrayLike] = None
    ) -> float:
        if weights is not None:
            self.build_weights(weights)
        predictions = model.predict(
            Snew=self.validation_data.S, Xnew=self.validation_data.X
        )
        targets = self.validation_data.Y
        losses = self.loss(targets, predictions)
        losses = np.squeeze(losses)
        return np.sum(losses * self.weights)


class BasicLossEstimator(WeightedEstimator):
    """
    A basic loss estimator that assigns equal weight to all validation data points.
    """

    def build_weights(self) -> float:
        return (
            np.ones(self.validation_data.S.shape[0]) / self.validation_data.S.shape[0]
        )

    def estimate_confidence(self, model: SpatialModel, use_cached=True) -> float:
        predictions = model.predict(
            Snew=self.validation_data.S, Xnew=self.validation_data.X
        )
        targets = self.validation_data.Y
        # Construct an unbiased estimate for variance of losses
        losses = self.loss(targets, predictions)
        losses = np.squeeze(losses)
        mean = np.mean(losses)
        var_loss = np.var(losses, ddof=1)
        # Estimate variance of sum
        var_sum = np.sum(self.weights**2 * var_loss)
        return np.sqrt(var_sum)


class RegressionEstimator(LossEstimator):
    """
    A class for estimating risk using regression.

    This class inherits from the LossEstimator class and provides an implementation
    of the estimate_risk method that uses a regression model to estimate the risk.

    """

    @abstractmethod
    def fit_regression(self, losses: ArrayLike) -> Tuple[ArrayLike, Dict]:
        """
        Fits a regression model to the given losses.

        Parameters:
        losses (ArrayLike): An array-like object containing the losses to fit the regression model to.

        Returns:
        ArrayLike: An array-like object containing the fitted regression model evaluated at test points.
        """
        pass

    def estimate_risk(self, model: SpatialModel):
        predictions = model.predict(
            Snew=self.validation_data.S, Xnew=self.validation_data.X
        )
        predictions = _to_np(predictions)
        targets = self.validation_data.Y
        losses = self.loss(targets, predictions)
        estimates = self.fit_regression(losses)[0]
        return np.average(estimates, axis=0)[0]


class NearestNeighborEstimator(RegressionEstimator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_neighbors = 1

    def neighbor_regressor(self):
        return KNeighborsRegressor(n_neighbors=1, algorithm="kd_tree")

    def fit_regression(self, losses: ArrayLike) -> ArrayLike:
        regressor = self.neighbor_regressor().fit(self.validation_data.S, losses)
        return regressor.predict(self.test_sites), 1


class HaversineNearestNeighborEstimator(NearestNeighborEstimator):

    def neighbor_regressor(self):
        return KNeighborsRegressor(
            n_neighbors=1, algorithm="ball_tree", metric="haversine"
        )


class FitNearestNeighborEstimator(NearestNeighborEstimator):

    def _calculate_k_bound(
        self, distances, inds, k, delta: Optional[float] = None, B: float = 1.0
    ):
        ntest = len(distances)
        # Calculate fill distance
        fill_distance = np.max(distances)
        # two norm
        _, counts = np.unique(inds, return_counts=True)
        counts_over_kntest = counts / k / ntest
        assert np.abs(np.sum(counts_over_kntest) - 1.0) < 1e-6
        two_norm = np.linalg.norm(counts_over_kntest)
        # constant
        const = B * np.sqrt(0.5 * np.log(2 / delta))
        return fill_distance + self.loss.upper_bound * const * two_norm

    def _find_n_neighbors(self, delta: Optional[float] = None):
        """
        Code currently assumes points are in [-0.5, 0.5]^d
        """
        if delta is None:
            # Rescale points to be in [0,1]^d
            pts = self.validation_data.S + 0.5
            good_inds = np.where(np.all((pts >= 0) & (pts <= 1), axis=1))[0]
            pts = pts[good_inds]
            # Keep track of max rescaling factor, will need to scale fill by this.
            # max_range = np.max(ran)
            # Compute approximate fill distance
            deltat = approximate_fill_distance(pts)
            # deltat = deltat * max_range
            delta = np.minimum(deltat, 1.0)
            print(f"Delta = {delta}")
        nval = len(self.validation_data.S)
        max_log2_k = int(np.log2(nval) // 1)  # floor(log_2(nval))
        possible_k = 2 ** np.arange(max_log2_k)  # (1, 2, ..., 2^floor(log_2(nval)))
        best_k = 1
        best_k_bound = np.inf
        nns = NearestNeighbors(algorithm="kd_tree")
        nns.fit(self.validation_data.S)
        for k in possible_k:
            # Run nearest neighbors
            distances, inds = nns.kneighbors(
                self.test_sites, n_neighbors=k, return_distance=True
            )
            # Calculate bound using neighbor weights and distances
            k_bound = self._calculate_k_bound(distances, inds, k, delta=delta)
            # update best k and bound
            if k_bound < best_k_bound:
                best_k = k
                best_k_bound = k_bound
        return best_k, best_k_bound

    def neighbor_regressor(self, delta: Optional[float] = None):
        n_neighbors, best_bound = self._find_n_neighbors(delta)
        self.num_neighbors = n_neighbors
        return KNeighborsRegressor(n_neighbors=n_neighbors, algorithm="kd_tree")


class HaversineFitNearestNeighborEstimator(FitNearestNeighborEstimator):

    def _find_n_neighbors(self):
        nval = len(self.validation_data.S)
        max_log2_k = int(np.log2(nval) // 1)  # floor(log_2(nval))
        possible_k = 2 ** np.arange(max_log2_k)  # (1, 2, ..., 2^floor(log_2(nval)))
        best_k = 1
        best_k_bound = np.inf
        nns = NearestNeighbors(algorithm="ball_tree", metric="haversine")
        nns.fit(self.validation_data.S)
        for k in possible_k:
            # Run nearest neighbors
            distances, inds = nns.kneighbors(
                self.test_sites, n_neighbors=k, return_distance=True
            )
            # Calculate bound using neighbor weights and distances
            k_bound = self._calculate_k_bound(distances, inds, k, delta=0.1)
            # update best k and bound
            if k_bound < best_k_bound:
                best_k = k
                best_k_bound = k_bound
        return best_k, best_k_bound

    def neighbor_regressor(self):
        n_neighbors, best_bound = self._find_n_neighbors()
        self.num_neighbors = n_neighbors
        return KNeighborsRegressor(
            n_neighbors=n_neighbors, algorithm="ball_tree", metric="haversine"
        )
