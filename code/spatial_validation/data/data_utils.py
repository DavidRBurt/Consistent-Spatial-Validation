from collections import namedtuple
from dataclasses import dataclass
from typing import Dict

import json_tricks
import numpy as np
from numpy.typing import ArrayLike
from tensorflow import Tensor

SpatialDataset = namedtuple(
    "SpatialDataset", ["S", "X", "Y"]
)  # Structure for storing spatial data
Dataset = namedtuple(
    "Dataset", ["train", "validation", "test"]
)  # Structure for storing data, all values should be arraylike


def dataset_to_dict(dataset: Dataset) -> Dict:
    data_dict = dict(
        train=dataset.train._asdict(),
        validation=dataset.validation._asdict(),
        test=dataset.test._asdict(),
    )
    return data_dict


def dict_to_dataset(data_dict: Dict, bootstrap: bool = False) -> Dataset:
    train = None if data_dict["train"] is None else SpatialDataset(**data_dict["train"])
    validation = (
        None
        if data_dict["validation"] is None
        else SpatialDataset(**data_dict["validation"])
    )
    test = None if data_dict["test"] is None else SpatialDataset(**data_dict["test"])
    if bootstrap:
        residuals = data_dict["residuals"]
        return BootstrappedSpatialDataset(train, validation, test, residuals)
    return Dataset(train, validation, test)


def load_dataset_from_json(fp: str, bootstrap: bool = False) -> Dataset:
    data_dict = json_tricks.load(fp)
    return dict_to_dataset(data_dict, bootstrap=bootstrap)


def ensure_numpy_dict(data_dict: Dict) -> Dict:
    new_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, Tensor):
            new_dict[k] = v.numpy()
        elif isinstance(v, dict):
            new_dict[k] = ensure_numpy_dict(v)
        else:
            new_dict[k] = v

    return new_dict


@dataclass
class BootstrappedSpatialDataset:
    train: SpatialDataset
    _validation: SpatialDataset
    _test: SpatialDataset
    _residuals: ArrayLike

    def bootstrap_spatial(self, rng, data):
        noise = rng.choice(self._residuals[:, 0], size=data.Y.shape[0], replace=True)
        return SpatialDataset(data.S, data.X, data.Y + noise[:, None])

    def __post_init__(self):
        # Called on init, always use same noise on training data, note this will lead to issues if same
        # seed is used here as on call because noise will be shared in a bad way
        rng2 = np.random.default_rng(seed=31415) 
        self.train = self.bootstrap_spatial(rng2, self.train)

    def validation(self, rng):
        return self.bootstrap_spatial(rng, self._validation)

    def test(self, rng):
        return self.bootstrap_spatial(rng, self._test)
    
    def __call__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        return Dataset(self.train, self.validation(rng), self.test(rng))
