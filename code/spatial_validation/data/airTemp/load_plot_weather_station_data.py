import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import plotly.graph_objects as go
import plotly.io as pio
import json_tricks
import urllib.request
import os
import certifi
import ssl


# turn off mathjax for plotly
pio.kaleido.scope.mathjax = None
LATEST = "v4.0.1.20240127" # Needs to be updated manually, can be found at https://www.ncei.noaa.gov/data/global-historical-climatology-network-monthly/v4/temperature/access/
DataDir = Path(Path(__file__).parent, "airTempData")
FigDir =  Path(Path(__file__).parents[4], "figures", "airTemp")
os.makedirs(FigDir, exist_ok=True)
os.makedirs(DataDir, exist_ok=True)

ssl._create_default_https_context = ssl._create_unverified_context

def download_weather_station_data():
    if not os.path.exists(str(Path(DataDir,f"ghcnm.tavg.{LATEST}.qcf.inv"))):
        url = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-monthly/v4/temperature/access/ghcnm.tavg.{LATEST}.qcf.inv"
        urllib.request.urlretrieve(url, str(Path(DataDir, f"ghcnm.tavg.{LATEST}.qcf.inv")))
    else:
        print(f"ghcnm.tavg.{LATEST}.qcf.inv already exists, skipping download")
    if not os.path.exists(str(Path(DataDir,f"ghcnm.tavg.{LATEST}.qcf.dat"))):
        url = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-monthly/v4/temperature/access/ghcnm.tavg.{LATEST}.qcf.dat"
        urllib.request.urlretrieve(url, str(Path(DataDir, f"ghcnm.tavg.{LATEST}.qcf.dat")))
    else:
        print(f"ghcnm.tavg.{LATEST}.qcf.dat already exists, skipping download")
    
    print("Downloaded weather station data.")

def read_station_id_info() -> Dict[str, Dict[str, str]]:
    with open(Path(str(Path(DataDir,f'ghcnm.tavg.{LATEST}.qcf.inv'))), 'r') as f:
        station_labels = f.readlines()
        station_labels = list(filter(lambda x: x.startswith("US"), station_labels))

    return build_station_dict_line(station_labels)

def build_station_dict_line(station_labels: List[str]):
    d = dict()
    for label in station_labels:
        ID = label[0:11].strip()
        lat = label[13:20].strip()
        long = label[21:30].strip()
        elevation = label[31:37].strip()
        name = label[41:71].strip()
        d[ID] = {
            "lat": lat,
            "long": long,
            "elevation": elevation,
            "name": name
        }
    return d

def read_temperature_data() -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
    with open(str(Path(DataDir, f'ghcnm.tavg.{LATEST}.qcf.dat')), 'r') as f:
        lines = f.readlines()
    # Filter out lines that do not start with US
    us_lines = filter(lambda x: x.startswith("US"), lines)
    # Split string into dictionary
    d = dict()
    for line in us_lines:
        ID = line[0:11].strip()
        year = line[11:15].strip()
        element = line[15:19].strip()
        if not element == "TAVG":
            continue
        if not ID in d:
            d[ID] = dict()
        if not year in d[ID]:
            d[ID][year] = dict()
        for i in range(0, 12):
            month = i + 1
            temperature = line[19 + i*8:24 + i*8].strip()
            dmflag = line[24 + i*8:25 + i*8].strip()
            qcflag = line[25 + i*8:26 + i*8].strip()
            dsflag = line[26 + i*8:27 + i*8].strip()
            d[ID][year][month] = {
                "temperature": temperature,
                "dmflag": dmflag,
                "qcflag": qcflag,
                "dsflag": dsflag
            }
    return d

def filter_by_year(d: Dict, year: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    d_year = dict()
    for ID, dp in d.items():
        if year in dp.keys():
            d_year[ID] = d[ID][year]
    return d_year

def filter_by_month(d: Dict, month: int) -> Dict[str, Dict[str, str]]:
    d_month = dict()
    for ID, dp in d.items():
        if month in dp.keys():
            d_month[ID] = d[ID][month]
    return d_month

def filter_quality_control(d: Dict) -> Dict[str, Dict[str, str]]:
    i =0
    new_d = dict()
    for ID, dp in d.items():
        if dp["qcflag"] != "" or dp["temperature"] == "-9999":
            i += 1
        else:
            new_d[ID] = dp
    print(f"Deleted {i} stations due to quality control flags or missing values.")
    return new_d


def get_lat_longs(data_dict: Dict):
    return [(float(d["lat"]), float(d["long"])) for _, d in data_dict.items()]

def combine_dicts(d1, d2):
    d = dict()
    for ID in d1.keys():
        if not ID in d2.keys():
            continue
        else:
            d[ID] = d1[ID]
            d[ID]["lat"] = d2[ID]["lat"]
            d[ID]["long"] = d2[ID]["long"]
            d[ID]["elevation"] = d2[ID]["elevation"]
            d[ID]["name"] = d2[ID]["name"]
    return d

def filter_alaska_and_hawaii(d):
    d_filtered = dict()
    for ID, dp in d.items():
        if float(dp["lat"]) < 23 or float(dp["lat"]) > 50:
            continue
        else:
            d_filtered[ID] = dp
    return d_filtered

if __name__ == "__main__":
    download_weather_station_data()
    year = '2018'
    month = 1
    station_dict = read_station_id_info()
    temperature_dict = read_temperature_data()
    temperature_dict_year = filter_by_year(temperature_dict, year)
    temperature_dict_month = filter_by_month(temperature_dict_year, month)
    temperature_dict_month = filter_quality_control(temperature_dict_month)

    data_dict = combine_dicts(temperature_dict_month, station_dict)
    data_dict = filter_alaska_and_hawaii(data_dict)


    fig = go.Figure(
    data=go.Scattergeo(
        lon=[float(d["long"]) for _, d in data_dict.items()],
        lat=[float(d["lat"]) for _, d in data_dict.items()],
        mode="markers",
        marker = dict(
            size=4,
            color=[float(d["temperature"])/100.0 for _, d in data_dict.items()],
            colorscale="Viridis",
            showscale=True,
        )
    )
)
    # center title, move closer to the ground, tighten margin
    fig.update_layout(
        title="Monthly Avg. Temp. at Weather Stations Jan 2018",
        title_x=0.5,
        title_y=0.9,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        geo_scope="usa",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
    )

    # fig.update_layout(coloraxis_colorbar=dict(title='Log10 County Population', tickprefix='1.e'))

    chorpleth = go.Choropleth(
        locationmode="USA-states",
        z=[0, 0],
        locations=["AK", "HI"],
        colorscale=[[0, "rgba(0, 0, 0, 0)"], [0, "rgba(0, 0, 0, 0)"]],
        marker_line_color="#fafafa",
        marker_line_width=2,
        showscale=False,
    )

    fig.add_trace(chorpleth)
    # save figure to pdf
    fig.write_image(str(Path(FigDir,"weather_stations.pdf")))
    # Save data dictionary to file
    json_tricks.dump(data_dict, str(Path(DataDir, "weather_stations.json")))
    
