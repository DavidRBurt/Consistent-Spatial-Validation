from typing import Dict, List, Optional
from time import time

import numpy as np
from numpy.typing import ArrayLike

import spatial_validation.estimators as estimators
import spatial_validation.models as models

from .data import Dataset, ensure_numpy_dict
from .losses import Loss, SquaredLoss, TruncatedSquaredLoss


def dict_to_str(d: Dict) -> str:
    s = ""
    for k, v in d.items():
        s += f"{k}={v}_"
    return s.rstrip("_")


class Experiment:
    def __init__(
        self,
        dataset: Dataset,
        model_names: List[str],
        model_parameters: List[Dict],
        param_names: List[str],
        estimator_names: List[str],
        estimator_parameters: List[Dict],
        loss: Loss,
    ) -> None:
        self.dataset = dataset
        self.loss = loss
        self.estimator_names = [
            name + "_" + dict_to_str(params)
            for name, params in zip(estimator_names, estimator_parameters)
        ]
        self.model_names = model_names
        self.estimators = self._build_estimators(estimator_names, estimator_parameters)
        self.models = self._build_models(model_names, model_parameters)
        self.model_parameters = model_parameters
        self.param_names = param_names

    def _build_estimators(
        self, estimator_names: List[str], estimator_parameters: List[Dict]
    ) -> List[estimators.LossEstimator]:
        """
        Construct all loss estimators to use in the experiment

        Args:
            estimator_names (List[str]): A list of methods to use for estimating loss, names must be
            classes in the "estimators.py"

        Returns:
            List[LossEstimator]: A list of estimators using the experiment data and names provided
        """
        return [
            getattr(estimators, name)(
                validation_data=self.dataset.validation,
                test_sites=self.dataset.test.S,
                loss=self.loss,
                **params,
            )
            for name, params in zip(estimator_names, estimator_parameters)
        ]

    def _build_models(
        self, model_names: List[str], model_parameters: List[Dict]
    ) -> List[models.SpatialModel]:
        return [
            getattr(models, name)(training_data=self.dataset.train, **params)
            for name, params in zip(model_names, model_parameters)
        ]

    def run(
        self,
        name: Optional[str] = None,
        has_ground_truth: bool = True,
    ) -> Dict:
        """

        Returns:
            Dict: [description]
        """
        results = dict()

        for i, (model, model_name, param_name) in enumerate(
            zip(self.models, self.model_names, self.param_names)
        ):
            name = "_".join([model_name, param_name])
            results[name] = dict()
            for est, est_name in zip(self.estimators, self.estimator_names):
                # Time how long risk estimation takes
                start_time = time()
                risk_estimate = est.estimate_risk(model)
                end_time = time()
                time_taken = end_time - start_time
                if isinstance(est, estimators.NearestNeighborEstimator):
                    est_params = int(est.num_neighbors)
                else:
                    est_params = None
                results[name][est_name] = dict(
                    estimate=risk_estimate, n_neighbors=est_params, time_taken=time_taken
                )
            if has_ground_truth:
                predictions = model.predict(self.dataset.test.S, self.dataset.test.X)
                sample_loss = np.mean(self.loss(self.dataset.test.Y, predictions))
                results[name]["sample_loss"] = sample_loss

        return ensure_numpy_dict(results)


class AirTempExperiment(Experiment):
    def run(
        self,
        name: Optional[str] = None,
    ) -> Dict:
        """

        Returns:
            Dict: [description]
        """
        results = dict()

        for i, (model, model_name, param_name) in enumerate(
            zip(self.models, self.model_names, self.param_names)
        ):
            name = "_".join([model_name, param_name])
            results[name] = dict()
            for est, est_name in zip(self.estimators, self.estimator_names):
                # If it is a BasicLossEstimator, loss confidence, otherwise confidence is None
                # time how long risk estimation takes
                start_time = time()
                risk_estimate = est.estimate_risk(model)
                end_time = time()
                time_taken = end_time - start_time
                estimate_confidence = None
                est_params = None
                if isinstance(est, estimators.BasicLossEstimator):
                    estimate_confidence = est.estimate_confidence(model)
                elif isinstance(est, estimators.NearestNeighborEstimator):
                    est_params = est.num_neighbors

                est_results = dict(
                    estimate=risk_estimate,
                    confidence=estimate_confidence,
                    est_params=est_params,
                    time_taken=time_taken,
                )
                results[name][est_name] = est_results

        return ensure_numpy_dict(results)
