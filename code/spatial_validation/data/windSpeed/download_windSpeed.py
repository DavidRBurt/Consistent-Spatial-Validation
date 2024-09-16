import urllib.request
from pathlib import Path
import os
import pandas as pd

# Download daily data from GHCN
def download_ghcn_data(datadir: Path) -> None:
    url = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/archive/daily-summaries-latest.tar.gz"
    tar_path = str(Path(datadir, "daily-summaries-latest.tar.gz"))
    extracted_path = str(Path(datadir, "daily-summaries-latest"))

    # Download tar file if not already downloaded
    if not os.path.exists(tar_path) and not os.path.exists(extracted_path):
        urllib.request.urlretrieve(url, tar_path)
    # Extract tar if not already extracted, then delete is
    if not os.path.exists(extracted_path):
        import tarfile
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extracted_path)
    if os.path.exists(tar_path):
        os.remove(tar_path)
    return extracted_path

def download_station_ids(datadir: Path) -> None:
    url = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
    station_path = str(Path(datadir, "ghcnd-stations.txt"))
    urllib.request.urlretrieve(url, station_path)

def filter_us_stations(extracteddir: Path) -> None:
    # Filter out only US stations, these should have an ID starting with US
    # loop over csv files in exracteddir and remove ones that don't start with US
    for file in extracteddir.iterdir():
        if file.suffix == ".csv":
            if file.stem[:2] != "US":
                os.remove(file)

def filter_contains_wind(extracteddir: Path) -> None:
    # Filter out only stations that contain wind data
    # loop over csv files in exracteddir and remove ones that don't contain wind
    for file in extracteddir.iterdir():
        if file.suffix == ".csv":
            with open(file, "r") as f:
                data = f.read()
                if "AWND" not in data:
                    os.remove(file)

def get_station_latlon_data(datadir:Path) -> pd.DataFrame:
    # load the station data
    station_path = str(Path(datadir, "ghcnd-stations.txt"))
    # read the station data line by line and split strings by character count
    stations = []
    with open(station_path, "r") as f:
        for line in f:
            stations.append(line)
    # split the strings by character count
    station_ids = [station[:11] for station in stations]
    station_lats = [float(station[12:20]) for station in stations]
    station_lons = [float(station[21:30]) for station in stations]
    return pd.DataFrame({"ID": station_ids, "Lat": station_lats, "Lon": station_lons})

def filter_conus(extracteddir: Path, station_latlon_data: pd.DataFrame) -> None:
    # Use lat and long to filter out stations that are not in the continental US
    # Continental US is defined as 24.396308 <= lat <= 49.384358 and -125.0 <= lon <= -66.93457
    for file in extracteddir.iterdir():
        if file.suffix == ".csv":
            station_id = file.stem
            station_lat = station_latlon_data[station_latlon_data.ID == station_id]["Lat"].values[0]
            station_lon = station_latlon_data[station_latlon_data.ID == station_id]["Lon"].values[0]
            if not (24.396308 <= station_lat <= 49.384358 and -125.0 <= station_lon <= -66.93457):
                os.remove(file)

def filter_nowindspeed(extracteddir: Path) -> None:
    # Filter out stations that don't have wind speed data
    for file in extracteddir.iterdir():
        if file.suffix == ".csv":
            # read with pandas
            data = pd.read_csv(file, low_memory=False)
            if "AWND" not in data.columns:
                os.remove(file)
                print("removing file")
            else:
                if data["AWND"].isnull().all():
                    os.remove(file)
                    print("removing file")

def plot_station_locations(extracteddir: Path, stations: pd.DataFrame) -> None:
    # Plot the locations of the stations
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    # convert these into a pandas dataframe
    # Check which station ids we have in the extracted directory
    wind_station_ids = [file.stem for file in extracteddir.iterdir() if file.suffix == ".csv"]
    stations = stations[stations.ID.isin(wind_station_ids)]
    # Plot the lat and lon of the stations on a map of the CONUS with some surrounding area
    gdf = gpd.GeoDataFrame(
        stations, geometry=gpd.points_from_xy(stations.Lon, stations.Lat)
    )
    gdf.plot(color="red")
    plt.savefig("station_locations.pdf")


if __name__ == "__main__":
    datadir = Path(Path(__file__).parent, "data")
    # Make the data directory if it doesn't exist
    os.makedirs(datadir, exist_ok=True)
    extracteddir = download_ghcn_data(str(datadir))
    download_station_ids(datadir)
    filter_us_stations(Path(extracteddir))
    filter_contains_wind(Path(extracteddir))
    # get station id -> lat, long data
    station_latlon_data = get_station_latlon_data(datadir)
    # filter continential US by lat and lon
    filter_conus(Path(extracteddir), station_latlon_data)
    # Plot data on world map to check we filtered things ok
    plot_station_locations(Path(extracteddir), station_latlon_data)
    # Filter out stations that don't have wind speed data
    # filter_nowindspeed(Path(extracteddir))
    # Load data into a giant pandas dataframe
    winddf = pd.concat([pd.read_csv(file, low_memory=False) for file in Path(extracteddir).iterdir() if file.suffix == ".csv"])
    # Drop rows that don't contain wind speed
    winddf = winddf.dropna(subset=["AWND"])
    # Select only the columns we need for the wind speed data. We will keep the data, the Lat, Lon, and the wind speed
    winddf = winddf[["STATION", "LATITUDE", "LONGITUDE", "DATE", "AWND"]]    
    # Save the data to a csv file
    winddf.to_csv(Path(datadir, "wind_speed_data.csv"), index=False)
    
