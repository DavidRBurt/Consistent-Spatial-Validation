## About
Code for the preprint "Consistent Validation for Predictive Methods in Spatial Settings" (David R. Burt, Yunyi Shen, Tamara Broderick), currently on ArXiv. 

## Package Installation
Navigate to the `code` directory and run `python -m pip install .` in a python 3.10 environment. The package requires tensorflow, so installation may take some time.

## Installation of Data
Most of the data is either downloaded or generated via scripts. The exception is the MODIS data. We downloaded this by navigating to 
https://opendap.cr.usgs.gov/opendap/hyrax/DP137/MOLA/MYD11C3.061/2018.01.01/MYD11C3.A2018001.061.2021319132947.hdf.dmr.html and selecting `get as netcdf4`. This requires an EarthData login, which can be registered for here: https://urs.earthdata.nasa.gov/users/new. The resulting file should be placed in the directory `code/spatial_validation/data/AirTemp/AirTempData`.

The GHCNM data is downloaded mostly automatically, but needs to be updated to point to the latest upload of the GHCNM archive, which can be found here: https://www.ncei.noaa.gov/data/global-historical-climatology-network-monthly/v4/temperature/access/. The variable `LATEST` in `spatial_validation/data/airTemp/load_plot_weather_station_data.py` line 16 should be of the form `v4.0.1.2024****`, where the stars are replaced by the most recent date data was updated.


## Running the code.
There is a `makefile` in the `code` directory. We provide a summary of commands that can be used to reproduce results below.

### Synthetic Experiments 

To generate the data run `make SyntheticDatasets`. This generates both datasets we investigated in the paper. It is currently setup to run 10 threads in parallel. This can be adjusted by modifying `code/experiments/synthetic/scripts/MakeSynthetic*Datasets.sh` to include a `-t NUMBER_OF_THREADS`.

To fit the model and estimate the risks run `make SyntheticResults`. The number of threads can again be adjusted by modifying `code/experiments/synthetic/scripts/RSynthetic*Experiment.sh` to include a `-t NUMBER_OF_THREADS`.

To generate plots run `make SyntheticGridPlots` and `make SyntheticPointPlots`. Plots will be generated in a `figures` directory.

### Synthetic Model Selection Experiment

To generate the data run `make ModelSelectionDatasets`.

To run the experiments first run `mkdir experiments/model_selection/results` to create a directory the results will be written to, then run `make ModelSelectionResults`.

To generate figures, run `make ModelSelectionPlots`.

### Air Temperature Experiment
After downloading the data needed (see the earlier section of the readme), remaining data can be downloaded a preprocessed by running: `make AirTempData`.

To run the experiments first run `mkdir experiments/airTemp/results` then run `make AirTempResults`.

To generate the tables run `python experiments/airTemp/plotting/make_table.py`.
