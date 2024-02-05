#!/bin/bash
ds="cluster"
echo "Plotting results for $ds"
TF_CPP_MIN_LOG_LEVEL=3 python experiments/synthetic/plotting/plot_consistency.py -d=$ds -nv 250 500 1000 2000 4000 8000