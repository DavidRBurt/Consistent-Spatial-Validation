#!/bin/bash
for nval in 125 250 500 1000 2000 4000 8000
do
    echo "Generating data for grid with $nval validation points" 
    CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python spatial_validation/data/synthetic/generate_data.py -d=cluster -nv=$nval --threads=10 --kernel=matern32
done
# CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python spatial_validation/data/synthetic/generate_data.py -d=cluster -nv=250 --plot --kernel=matern32
