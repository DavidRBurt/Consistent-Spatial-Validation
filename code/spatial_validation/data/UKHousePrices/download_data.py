import os
from pathlib import Path
import json_tricks
import urllib.request
import pandas as pd
import numpy as np


datadir = Path(Path(__file__).parent, "data")
os.makedirs(str(datadir), exist_ok=True)

download_path = Path(datadir, "UK-house-prices-2023.csv")
ukzip_path = Path(datadir, "combined_ukzip.csv")
if not os.path.exists(str(download_path)):
    urllib.request.urlretrieve(
        "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2023.csv",
        str(download_path),
    )

postal_df = pd.read_csv(str(ukzip_path))

housing_df = pd.read_csv(str(download_path))
# https://www.gov.uk/guidance/about-the-price-paid-data#explanations-of-column-headers-in-the-ppd
housing_df.columns = [
    "UID",
    "PRICE",
    "DATE",
    "POSTCODE",
    "TYPE",
    "NEW",
    "DURATION",
    "PAON",
    "SAON",
    "STREET",
    "LOCALITY",
    "CITY",
    "DISTRICT",
    "COUNTY",
    "PPDTYPE",
    "RECORDSTATUS",
]
# Only consider additions to housing dataset
housing_df = housing_df.loc[housing_df.RECORDSTATUS == "A"]
# Only use standard price paid
housing_df = housing_df.loc[housing_df.PPDTYPE == "A"]
# Only consider flats following Hensman, Fusi, Lawrence paper, might change this later.
housing_df = housing_df.loc[housing_df.TYPE == "F"]
# Build dictionary to lookup postcodes
postal_lat_lookup_dict = {
    pc: lat
    for pc, lat in zip(postal_df.PC, postal_df.LAT)
}
postal_long_lookup_dict = {
    pc: long
    for pc, long in zip(postal_df.PC, postal_df.LONG)
}
housing_df["LAT"] = housing_df["POSTCODE"].map(postal_lat_lookup_dict)
housing_df["LONG"] = housing_df["POSTCODE"].map(postal_long_lookup_dict)
housing_df["LOGPRICE"] = housing_df["PRICE"].map(lambda x: np.log10(x))
housing_df = housing_df.dropna(subset=["LAT"])
housing_df = housing_df.dropna(subset=["LONG"])


london_housing_df = (housing_df.loc[housing_df.CITY == "LONDON"])[["LOGPRICE", "LAT", "LONG"]]
nonlondon_housing_df = (housing_df.loc[housing_df.CITY != "LONDON"])[["LOGPRICE", "LAT", "LONG"]]

datadict = dict(
    london=london_housing_df.to_numpy(),
    nonlondon=nonlondon_housing_df.to_numpy()
)
savepath = Path(datadir, "UKHousingData.json")
json_tricks.dump(datadict, str(savepath))
