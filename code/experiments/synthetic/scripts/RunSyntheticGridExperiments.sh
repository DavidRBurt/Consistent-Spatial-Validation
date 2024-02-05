#!/bin/bash
for nval in 125 250 500 1000 2000 4000 8000
do
    echo "Running Experiment for cluster with $nval validation points"
    CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 python experiments/synthetic/run_experiment.py -d=cluster -nv=$nval
done