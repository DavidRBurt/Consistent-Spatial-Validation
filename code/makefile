# Directories
syn_data_dir := spatial_validation/data/synthetic
syn_exp_dir := experiments/synthetic
syn_scripts_dir := experiments/synthetic/scripts

# These will be empty files created/dated with when experiment was run to avoid unnecessary reruns
SyntheticGridDatasetsComplete = $(syn_data_dir)/data/grid_complete.log
SyntheticGridResultsComplete = $(syn_exp_dir)/results/grid_complete.log

# Aliases
SyntheticGridDatasets: ${SyntheticGridDatasetsComplete}
SyntheticGridResults: ${SyntheticGridResultsComplete}
.Phony : SyntheticGridPlots # Plots are always regenerated, this is fast though.

# Commands for running synthetic experiments.
${SyntheticGridDatasetsComplete}: $(syn_data_dir)/generate_data.py
	bash $(syn_scripts_dir)/MakeSyntheticGridDatasets.sh >  $(syn_data_dir)/data/grid_out.log
	touch ${SyntheticGridDatasetsComplete}

${SyntheticGridResultsComplete}: ${SyntheticGridDatasetsComplete} $(syn_exp_dir)/run_experiment.py
	bash $(syn_scripts_dir)/RunSyntheticGridExperiments.sh > $(syn_exp_dir)/results/grid_out.log
	touch ${SyntheticGridResultsComplete}

SyntheticGridPlots: ${SyntheticGridResultsComplete} $(syn_exp_dir)/plotting/plot_consistency.py
	bash $(syn_scripts_dir)/PlotSyntheticGridExperiments.sh

# These will be empty files created/dated with when experiment was run to avoid unnecessary reruns
SyntheticPointDatasetsComplete = $(syn_data_dir)/data/point_complete.log
SyntheticPointResultsComplete = $(syn_exp_dir)/results/point_complete.log
# Aliases for readability
SyntheticPointDatasets: ${SyntheticPointDatasetsComplete}
SyntheticPointResults: ${SyntheticPointResultsComplete}
.Phony : SyntheticPointPlots # Plots are always regenerated, this is fast though.

# Commands for running synthetic experiments.
${SyntheticPointDatasetsComplete}: $(syn_data_dir)/generate_data.py
	bash $(syn_scripts_dir)/MakeSyntheticPointDatasets.sh >  $(syn_data_dir)/data/point_out.log
	touch ${SyntheticPointDatasetsComplete}

${SyntheticPointResultsComplete}: ${SyntheticPointDatasetsComplete} $(syn_exp_dir)/run_experiment.py
	bash $(syn_scripts_dir)/RunSyntheticPointExperiments.sh > $(syn_exp_dir)/results/point_out.log
	touch ${SyntheticPointResultsComplete}

SyntheticDatasets: SyntheticPointDatasets SyntheticGridDatasets
SyntheticResults: SyntheticPointResults SyntheticGridResults



# Model Selection Experiment 

# Directories
ms_data_dir := spatial_validation/data/model_selection
ms_exp_dir := experiments/model_selection
ms_scripts_dir := experiments/model_selection/scripts

# These will be empty files created/dated with when experiment was run to avoid unnecessary reruns
ModelSelectionDatasetsComplete = $(ms_data_dir)/complete.log
ModelSelectionResultsComplete = $(ms_exp_dir)/results/complete.log

# Aliases for readability
ModelSelectionDatasets: ${ModelSelectionDatasetsComplete}
ModelSelectionResults: ${ModelSelectionResultsComplete}
.Phony : ModelSelectionPlots # Plots are always regenerated, this is fast though.

# Commands for running synthetic experiments.
${ModelSelectionDatasetsComplete}: $(ms_data_dir)/generate_data.py
	bash $(ms_scripts_dir)/MakeModelSelectionDatasets.sh >  $(ms_data_dir)/out.log
	touch ${ModelSelectionDatasetsComplete}

${ModelSelectionResultsComplete}: ${ModelSelectionDatasetsComplete} $(ms_exp_dir)/model_selection.py
	bash $(ms_scripts_dir)/RunModelSelectionExperiment.sh > $(ms_exp_dir)/results/out.log
	touch ${ModelSelectionResultsComplete}

# Didn't bother to script this, just call python
ModelSelectionPlots: ${ModelSelectionResultsComplete} $(ms_exp_dir)/plotting/plot_model_selection.py
	python $(ms_exp_dir)/plotting/plot_model_selection.py 


at_data_dir := spatial_validation/data/airTemp
at_exp_dir := experiments/airTemp/
.Phony : AirTempData
.Phony : AirTempResults

# Commands for downloading and preprocessing air temperature experiment

AirTempData: $(at_data_dir)/AirTempData/metro_temperature_data.json $(at_data_dir)/AirTempData/grid_temperature_data.json

$(at_data_dir)/AirTempData/grid_temperature_data.json: $(at_data_dir)/build_dataset.py $(at_data_dir)/AirTempData/weather_stations.json
	python spatial_validation/data/airTemp/build_dataset.py

$(at_data_dir)/AirTempData/metro_temperature_data.json: $(at_data_dir)/build_dataset.py $(at_data_dir)/AirTempData/weather_stations.json
	python spatial_validation/data/airTemp/build_dataset.py

$(at_data_dir)/AirTempData/weather_stations.json: $(at_data_dir)/load_plot_weather_station_data.py
	python spatial_validation/data/airTemp/load_plot_weather_station_data.py

# Commands for running air temperature experiment
AirTempResults: $(at_exp_dir)/results/gwr-grid.json $(at_exp_dir)/results/gwr-metro.json

$(at_exp_dir)/results/gwr-metro.json: AirTempData experiments/airTemp/airtemp_experiment.py $(at_data_dir)/AirTempData/metro_temperature_data.json
	python experiments/airTemp/airtemp_experiment.py -d metro

$(at_exp_dir)/results/gwr-grid.json: AirTempData experiments/airTemp/airtemp_experiment.py $(at_data_dir)/AirTempData/grid_temperature_data.json
	python experiments/airTemp/airtemp_experiment.py -d grid

	

