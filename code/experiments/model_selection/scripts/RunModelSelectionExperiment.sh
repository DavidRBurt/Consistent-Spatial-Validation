#!/bin/bash
for nval in 5 15 25 35 45 55 65 75
do
    echo "Running Experiment for Model Selection with $nval validation points"
    CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python experiments/model_selection/model_selection.py -nv=$nval
done