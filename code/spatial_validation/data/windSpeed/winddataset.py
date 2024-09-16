from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from spatial_validation.data import SpatialDataset, Dataset


class WindDataset:

    def __init__(self):
        datapath = Path(Path(__file__).parent, "data", "wind_speed_data.csv")
        self._data = pd.read_csv(datapath)
        # Drop rows with date before 2000
        self._data["YEAR"] = self._data["DATE"].apply(lambda x: int(str(x)[:4]))
        self._data = self._data[self._data["YEAR"] >= 2000]
        # Keep rows in January only
        self._data["MONTH"] = self._data["DATE"].apply(lambda x: int(str(x)[5:7]))
        self._data = self._data[self._data["MONTH"] == 1]

        self._unique_stations = self._data["STATION"].unique()
        self._test_station_id = "USW00094846"
        self._test_station_data = self._data[
            self._data["STATION"] == self._test_station_id
        ]
        # Remove test station from the data
        self._unique_stations = self._unique_stations[
            self._unique_stations != self._test_station_id
        ]

    def _split_train_validation(
        self,
        unique_stations: ArrayLike,
        rng: np.random.RandomState,
    ) -> Tuple[ArrayLike, ArrayLike]:
        # split the unique stations into train and validation sets
        rng.shuffle(unique_stations)
        split_index = int(0.8 * len(unique_stations))
        return unique_stations[:split_index], unique_stations[split_index:]

    def pd_to_sd(
        self,
        data: pd.DataFrame,
        rng: Optional[np.random.RandomState] = None,
        perturb: bool = False,
    ) -> SpatialDataset:
        S = data[["LATITUDE", "LONGITUDE"]].to_numpy()
        if perturb:
            if rng is None:
                raise ValueError("rng must be provided if perturb is True")
            S += 1e-12 * rng.randn(len(data), 2)
        Y = data["AWND"].to_numpy()[:, None]
        return SpatialDataset(S=S, X=None, Y=Y)

    def __call__(self, seed: int) -> Dataset:
        rng = np.random.RandomState(seed)
        train_station_ids, validation_station_ids = self._split_train_validation(
            self._unique_stations, rng
        )
        train_data = self._data[self._data["STATION"].isin(train_station_ids)]
        validation_data = self._data[self._data["STATION"].isin(validation_station_ids)]
        return Dataset(
            train=self.pd_to_sd(train_data),
            validation=self.pd_to_sd(validation_data, rng=rng, perturb=True),
            test=self.pd_to_sd(self._test_station_data),
        )

if __name__ == "__main__":
    ds = WindDataset()
