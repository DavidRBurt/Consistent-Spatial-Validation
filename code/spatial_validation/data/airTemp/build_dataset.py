import json_tricks
import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import radians
import os
from pathlib import Path
import xarray as xr
import reverse_geocoder as rg
import requests, zipfile, io


DataDir = Path(Path(__file__).parent, "airTempData")
dp = lambda f: str(Path(DataDir, f))  # datapath


def download_ub_data():
    target_url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_ua_national.zip"
    r = requests.get(target_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dp("urban_centers"))
    geoids = [63217, 51445, 16264, 56602, 40429]  # NYC, LA, Chi, Miami, Houston
    datapath = dp("urban_centers/2023_Gaz_ua_national.txt")
    ub = np.loadtxt(
        datapath, delimiter="\t", dtype=float, skiprows=1, usecols=[0, 7, 8]
    )
    fn = lambda x: int(x[0]) in [63217, 51445, 16264, 56602, 40429]
    metro_lat_lons = np.stack([m[1:] for m in filter(fn, ub)], axis=0)
    return metro_lat_lons


def download_lst_data():
    if not os.path.exists(dp("MYD11C3.A2018001.061.2021319132947.hdf.nc4")):
        # can download manually from https://opendap.cr.usgs.gov/opendap/hyrax/DP137/MOLA/MYD11C3.061/2018.01.01/MYD11C3.A2018001.061.2021319132947.hdf.dmr.html
        raise FileNotFoundError(
            "Not Currently Setup to automatically download LST data"
        )


def subset_lst_data():
    if not os.path.exists("LST.json"):
        fn = dp("MYD11C3.A2018001.061.2021319132947.hdf.nc4")
        ds = xr.open_dataset(fn)
        # Subset netcdf file to roughly Continental US
        ds = ds.sel(Latitude=slice(50, 25), Longitude=slice(-125, -65))
        # Remove all values where LST_Day_CMG is zero (no data)
        ds = ds.where(ds.LST_Day_CMG != 0)
        # Remove all values where LST_Night_CMG is zero (no data)
        ds = ds.where(ds.LST_Night_CMG != 0)
        # Extract LST_Day_CMG and LST_Night_CMG
        ds = ds[["LST_Day_CMG", "LST_Night_CMG"]]
        # Convert Kelvin to Celsius
        ds["LST_Day_CMG"] = ds["LST_Day_CMG"] - 273.15
        ds["LST_Night_CMG"] = ds["LST_Night_CMG"] - 273.15
        # Convert ds to a pandas dataframe
        ds = ds.to_dataframe()
        # Remove all rows where LST_Day_CMG is NaN
        ds = ds.dropna(subset=["LST_Day_CMG"])
        # Remove all rows where LST_Night_CMG is NaN
        ds = ds.dropna(subset=["LST_Night_CMG"])

        # Get lat long index and convert to numpy
        inds = np.array(list(ds.index.to_numpy()))
        vals = ds.to_numpy()
        # Save to json
        json_tricks.dump(dict(S=inds, X=vals), dp("LST.json"))
    else:
        print("LST.json already exists, skipping parsing")


if __name__ == "__main__":
    # set numpy seed
    rng = np.random.default_rng(seed=1)
    # Download LST data
    download_lst_data()
    # Subset LST data
    subset_lst_data()
    # urban areas data
    metro_lat_lons = download_ub_data()

    # Load the weather station data
    stations_data_dict = json_tricks.load(dp("weather_stations.json"))
    # Latitude, longitude of each station in Radians
    station_lat_lons = np.array(
        [
            (radians(float(d["lat"])), radians(float(d["long"])))
            for _, d in stations_data_dict.items()
        ]
    )
    # Temperature of each station in degrees C
    station_temps = (
        np.array([d["temperature"] for _, d in stations_data_dict.items()]).astype(
            float
        )
        / 100.0
    )
    # Load the LST data
    lst_data_dict = json_tricks.load(dp("LST.json"))
    # Latitude, longitude of each LST point in Radians
    lst_lat_lons = np.array(
        [(radians(s[0]), radians(s[1])) for s in lst_data_dict["S"]]
    )
    # Temperature of each LST point in degrees C
    lst = np.array(lst_data_dict["X"])
    # match the weather station data to the closest LST point
    neigh = NearestNeighbors(n_neighbors=1, metric="haversine", algorithm="ball_tree")
    neighbor_fit = neigh.fit(lst_lat_lons)
    distances, inds = neighbor_fit.kneighbors(station_lat_lons)
    inds = np.squeeze(inds)
    # Get the stations more than 4km from the closest LST point, remove these from training set
    bad_inds = np.where(distances * 6371 > 4.0)
    # # Remove the bad indices from the station data
    # station_lat_lons = np.delete(station_lat_lons, bad_inds, axis=0)
    # inds = np.delete(inds, bad_inds)
    # station_temps = np.delete(station_temps, bad_inds, axis=0)
    # get the LST data at the closest points
    S = station_lat_lons
    X = lst[inds]
    Y = station_temps
    # Split off a random 70% of the data for validation
    n = X.shape[0]
    n_train = int(n * 0.7)
    train_indices = rng.choice(n, n_train, replace=False)
    val_indices = np.setdiff1d(np.arange(n), train_indices)
    S_train = S[train_indices]
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    S_val = S[val_indices]
    X_val = X[val_indices]
    Y_val = Y[val_indices]
    train_data = {"S": S_train, "X": X_train, "Y": Y_train[:, None]}
    val_data = {"S": S_val, "X": X_val, "Y": Y_val[:, None]}

    # Remove points outside USA using reverse_geocoder, this is a bit messy
    lst_lat_lon_degrees = [(np.degrees(s[0]), np.degrees(s[1])) for s in lst_lat_lons]

    # Get the country code for each point
    results = rg.search(lst_lat_lon_degrees)
    # Only keep points inside USA, the boundary is a bit crude here.
    in_US = [result["cc"] == "US" for result in results]
    lst_lat_lons = lst_lat_lons[in_US]
    lst = lst[in_US]
    # Grid prediction data
    data_dict = dict(
        train=train_data,
        validation=val_data,
        test={"S": lst_lat_lons, "X": lst, "Y": None},
    )

    json_tricks.dump(data_dict, dp("grid_temperature_data.json"))
    metro_lat_lons = np.array([(radians(s[0]), radians(s[1])) for s in metro_lat_lons])
    #
    distances, inds = neighbor_fit.kneighbors(metro_lat_lons)
    inds = np.squeeze(inds)
    lst = np.array(lst_data_dict["X"])
    metro_lsts = lst[inds]
    data_dict = dict(
        train=train_data,
        validation=val_data,
        test={"S": metro_lat_lons, "X": metro_lsts, "Y": None},
    )
    json_tricks.dump(data_dict, dp("metro_temperature_data.json"))
