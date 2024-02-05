#!/bin/bash
for nval in 250 500 1000 2000 4000 8000
do
    echo "Running Experiment for point_prediction with $nval validation points"
    CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python experiments/synthetic/run_experiment.py -d=point_prediction -nv=$nval --rbf
done