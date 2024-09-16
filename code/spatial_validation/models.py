from abc import abstractmethod
from typing import Tuple, Callable, List, Optional

import numpy as np
import gpflow
from gpflow.kernels import Kernel
from gpflow.models import GPModel
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics.pairwise import haversine_distances
from numpy.typing import ArrayLike
import tensorflow as tf
from lightgbm import Dataset as lgbDataset
from lightgbm import LGBMRegressor

from spatial_validation.data import SpatialDataset
from gpflow.models import GPR, SGPR
from gpflow.utilities import add_likelihood_noise_cov
from gpflow.optimizers import Scipy

errstatus = np.geterr()
from robustgp.init_methods import ConditionalVariance
np.seterr(**errstatus)


class KRR(GPR):

    def predict_f(self, Xnew):
        # Overwrite predict method to solve fast direction

        X, Y = self.data
        err = Y - self.mean_function(X)

        kmm = self.kernel(X)
        knm = self.kernel(Xnew, X)  # [s, n]
        kmm_plus_s = add_likelihood_noise_cov(kmm, self.likelihood, X)
        L = tf.linalg.cholesky(kmm_plus_s)
        Kinvy = tf.linalg.cholesky_solve(L, Y)  # [n, 1]
        return tf.linalg.matmul(knm, Kinvy)


class SpatialModel:
    def __init__(
        self,
        training_data: SpatialDataset,
    ):
        self._data = training_data

    @property
    def S(self) -> ArrayLike:
        """
        Spatial Coordinates

        Returns:
            Arraylike: Coordinates, [N, D]
        """
        return self._data.S

    @property
    def X(self) -> ArrayLike:
        """
        Covariates

        Returns:
            Arraylike: Covariates, [N, P]
        """
        return self._data.X

    @property
    def Y(self) -> ArrayLike:
        """
        Response Variable

        Returns:
            ArrayLike: Response, [N, 1]
        """
        return self._data.Y

    @property
    def num_train(self) -> int:
        return self.Y.shape[0]

    @abstractmethod
    def predict(self, Snew: ArrayLike, Xnew: ArrayLike) -> ArrayLike:
        """
        Args:
            Snew (ArrayLike): [description]
            Xnew (ArrayLike): [description]

        Returns:
            ArrayLike: [description]
        """


class ConstantPredictor(SpatialModel):
    def __init__(
        self,
        training_data: SpatialDataset,
        value: float,
    ):
        super().__init__(training_data=training_data)
        self.value = value

    def predict(self, Snew: ArrayLike, Xnew: ArrayLike) -> ArrayLike:
        return self.value * np.ones((len(Snew), 1))


class MAELinearModel(SpatialModel):
    def __init__(
        self,
        training_data: SpatialDataset,
    ):
        super().__init__(training_data=training_data)
        self._model = self._build_model()

    def _build_model(self):
        m = QuantileRegressor(alpha=0.0, fit_intercept=True, solver="highs")
        m.fit(self.S, self.Y[:, 0])
        return m

    def predict(self, Snew: ArrayLike, Xnew: ArrayLike) -> ArrayLike:
        return self._model.predict(Snew)[:, None]


class KRRSpatialRegression(SpatialModel):
    def __init__(
        self,
        training_data: Tuple[ArrayLike],
        response_kernel: Kernel,
        regularization_parameter: float,
        center=False,
        max_likelihood: bool = False,
    ) -> None:
        """
        Args:
            data (Tuple[ArrayLike]): (S, X, Y),
                S is [N, D] and consists of spatio-temporal coordinates.
                X = X(S) is [N, P] and consists of covariate values.
                Y = Y(S) is [N, 1] and consists of response values.

            covariate_kernels List[Kernel]: Currently not used
                List of P kernels to used in modelling spatial covariates.
                When performing FFanova, we assume X_p ~ GP(0, k_p).

            response_kernel Kernel: [description].
        """
        super().__init__(training_data)
        self.response_mean = np.mean(self.Y) if center else 0.0
        self.response_kernel = response_kernel
        self._gpr_model = self._build_model(
            regularization_parameter, max_likelihood=max_likelihood
        )

    def _build_model(
        self, regularization_parameter: float, max_likelihood=False
    ) -> GPR:
        stacked_data = tf.concat([self.X, self.S], axis=-1)
        stacked_data = tf.cast(stacked_data, tf.float64)
        Y = tf.cast(self.Y, tf.float64)
        model = KRR(
            data=(stacked_data, Y),
            kernel=self.response_kernel,
            mean_function=gpflow.mean_functions.Constant(c=self.response_mean),
            noise_variance=regularization_parameter,
        )
        # Optimize model hyperparameters with maximum marginal likelihood
        if max_likelihood:
            opt = Scipy()
            gpflow.set_trainable(model.mean_function, False)
            opt.minimize(
                model.training_loss_closure(),
                variables=model.trainable_variables,
                options=dict(maxiter=15, disp=False),
            )
        return model

    def predict(self, Snew: ArrayLike, Xnew: ArrayLike) -> tf.Tensor:
        if Xnew is not None:
            all_new = tf.concat([Xnew, Snew], axis=-1)
            all_new = tf.cast(all_new, tf.float64)
        else:
            all_new = Snew

        return self._gpr_model.predict_f(all_new)


class LowRankKRRSpatialRegression(KRRSpatialRegression):
    def __init__(
        self,
        training_data: Tuple[ArrayLike],
        response_kernel: Kernel,
        regularization_parameter: float,
        num_inducing: int,
        center=False,
        max_likelihood: bool = False,
    ) -> None:
        self.num_inducing = num_inducing
        super().__init__(
            training_data,
            response_kernel,
            regularization_parameter,
            center,
            max_likelihood,
        )

    def init_inducing(self, covariates):
        init_method = ConditionalVariance()
        Z, _ = init_method.compute_initialisation(
            covariates.numpy(), self.num_inducing, self.response_kernel
        )
        return Z

    def _build_model(
        self, regularization_parameter: float, max_likelihood=False
    ) -> SGPR:
        if self.X is not None:
            stacked_data = tf.concat([self.X, self.S], axis=-1)
        else:
            stacked_data = self.S
        stacked_data = tf.cast(stacked_data, tf.float64)
        Y = tf.cast(self.Y, tf.float64)
        Z = self.init_inducing(stacked_data)
        model = SGPR(
            data=(stacked_data, Y),
            kernel=self.response_kernel,
            mean_function=gpflow.mean_functions.Constant(c=self.response_mean),
            noise_variance=regularization_parameter,
            inducing_variable=Z,
        )
        # Optimize model hyperparameters with maximum marginal likelihood
        if max_likelihood:
            opt = Scipy()
            # gpflow.set_trainable(model.mean_function, False)
            opt.minimize(
                model.training_loss_closure(),
                variables=model.trainable_variables,
                options=dict(maxiter=15, disp=True),
            )
        return model

    def predict(self, Snew: ArrayLike, Xnew: ArrayLike) -> tf.Tensor:
        if Xnew is not None:
            all_new = tf.concat([Xnew, Snew], axis=-1)
            all_new = tf.cast(all_new, tf.float64)
        else:
            all_new = Snew

        return self._gpr_model.predict_f(all_new)[0]


def default_weight_fn(S: ArrayLike, Snew: ArrayLike, lengthscale: float) -> ArrayLike:
    """
    Args:
        S (ArrayLike): [description]
        Snew (ArrayLike): [description]

    Returns:
        ArrayLike: [description]
    """
    distances = haversine_distances(S, Snew) * 6371
    weights = np.exp(-(distances**2) / lengthscale**2)
    return weights


class GeographicallyWeightedRegression(SpatialModel):

    def __init__(
        self,
        training_data: SpatialDataset,
        weight_function: Callable = default_weight_fn,
    ):
        super().__init__(training_data)
        self._weight_function = lambda x, lss: weight_function(
            x, self.S, lss
        )  # partial fn application
        self.lengthscale = None

        self.model = self._build_model()

    def _build_model(self) -> LinearRegression:
        model = LinearRegression()
        self._fit_lengthscale_via_cv(
            [25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 750.0, 1000.0]
        )
        model.fit(self.X, self.Y[:, 0])
        return model  # This is a place holder, shouldn't really be used

    def _fit_lengthscale_via_cv(self, lengthscales: List[float]) -> float:
        """
        Fit the lengthscale parameter via LOO-CV
        """
        model = LinearRegression()
        cur_best_lss = lengthscales[0]
        cur_best_loss = np.inf
        # We probably will want to parallelize all this, at least over lengthscales
        for lengthscale in lengthscales:
            losses = list()
            for i, (s, x, y) in enumerate(zip(self.S, self.X, self.Y)):
                del_row = lambda arr: np.delete(arr, i, axis=0)
                weights = self._weight_function(s[None, :], lengthscale)
                # Fit the model leaving out the ith row
                model.fit(
                    del_row(self.X),
                    del_row(self.Y[:, 0]),
                    sample_weight=del_row(weights[0]),
                )
                # Make predictions and compute loss on the ith row
                predictions = model.predict(x[None, :])
                loss = np.square(y - predictions)
                losses.append(loss)
            # Compute the mean loss
            mean_loss = np.mean(losses)
            # Update best lengthscale
            if mean_loss < cur_best_loss:
                cur_best_loss = mean_loss
                cur_best_lss = lengthscale
        self.lengthscale = cur_best_lss
        print(f"Chosen lengthscale: {cur_best_lss} with loss {cur_best_loss}")

    def predict(self, Snew: ArrayLike, Xnew: ArrayLike):
        model = LinearRegression()
        predictions = []
        for s, x in zip(Snew, Xnew):
            weights = self._weight_function(s[None, :], self.lengthscale)
            model.fit(self.X, self.Y[:, 0], sample_weight=weights[0])
            predictions.append(model.predict(x[None, :]))

        return np.concatenate(predictions, axis=0)[:, None]


class GBMSpatialModel(SpatialModel):
    def __init__(
        self,
        training_data: SpatialDataset,
        num_leaves: int,
        n_estimators: int,
    ):
        super().__init__(training_data)
        self.model = self._build_model(num_leaves, n_estimators)

    def _build_model(self, num_leaves: int, n_estimators: int) -> LGBMRegressor:
        model = LGBMRegressor(
            num_leaves=num_leaves,
            n_estimators=n_estimators,
        )
        if self.X is not None:
            stacked_data = np.concatenate([self.X, self.S], axis=-1)
        else:
            stacked_data = self.S

        model.fit(stacked_data, self.Y[:, 0])
        return model

    def predict(self, Snew: ArrayLike, Xnew: Optional[ArrayLike] = None) -> ArrayLike:
        if Xnew is not None:
            stacked_data = np.concatenate([Xnew, Snew], axis=-1)
        else:
            stacked_data = Snew

        return self.model.predict(stacked_data)[:, None]
