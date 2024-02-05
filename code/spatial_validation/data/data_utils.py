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


def dict_to_dataset(data_dict: Dict) -> Dataset:
    train = SpatialDataset(**data_dict["train"])
    validation = SpatialDataset(**data_dict["validation"])
    test = SpatialDataset(**data_dict["test"])
    return Dataset(train, validation, test)


def load_dataset_from_json(fp: str) -> Dataset:
    data_dict = json_tricks.load(fp)
    return dict_to_dataset(data_dict)


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
