from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import json_tricks

from spatial_validation.data import SpatialDataset, Dataset


class UKHousingDataset:
    def __init__(self, ntest: int = 1000, ntrain: int = 40000):
        self.london_houses, self.nonlondon_houses = self.load_ds()
        self.ntrain = ntrain
        self.ntest = ntest

    def load_ds(self):
        dspath = str(Path(Path(__file__).parent, "data", "UKHousingData.json"))
        datadict = json_tricks.load(dspath)
        return datadict["london"], datadict["nonlondon"]

    def __call__(self, seed: int = 0) -> Dataset:

        rng = np.random.default_rng(seed)
        # Shuffle and get indices
        london_shuffle = rng.permutation(range(len(self.london_houses)))
        nonlondon_shuffle = rng.permutation(range(len(self.nonlondon_houses)))
        test_inds = london_shuffle[: self.ntest]
        london_val_inds = london_shuffle[self.ntest :]
        train_inds = nonlondon_shuffle[: self.ntrain]
        nonlondon_val_inds = nonlondon_shuffle[self.ntrain :]
        # Slice into dataframes and build datasets
        train_df = self.nonlondon_houses[train_inds]
        test_df = self.london_houses[test_inds]
        valnonlondon_df = self.nonlondon_houses[nonlondon_val_inds]
        vallondon_df = self.london_houses[london_val_inds]
        val_df = np.concatenate([valnonlondon_df, vallondon_df], axis=0)
        # Convert to expected dataset type
        return Dataset(self._sd(train_df), self._sd(val_df), self._sd(test_df))

    def _sd(self, df: ArrayLike) -> SpatialDataset:
        return SpatialDataset(S=df[:, 1:], X=None, Y=df[:, 0:1])
