#!/bin/bash

set -e  # Stop immediately if any command fails

echo "[1/3] Running Part 2: Linear Regression Double Descent..."
python /home/jupyter-iec2024iot03/CS115/src/part2_linear_exp.py

echo
echo "[2/3] Running Part 3: RFF Experiment on MNIST..."
python /home/jupyter-iec2024iot03/CS115/src/part3_rff_exp.py

echo
echo "[3/3] Plotting results..."
python /home/jupyter-iec2024iot03/CS115/src/plot_results.py

echo
echo "All experiments completed successfully."
echo "Figures saved in the 'plots/' directory."
