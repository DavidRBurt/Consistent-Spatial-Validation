#!/bin/bash
for nval in 5 15 25 35 45 55 65 75
do
    echo "Generating data for model selection with $nval validation points" 
    CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python spatial_validation/data/model_selection/generate_data.py -nv=$nval
done