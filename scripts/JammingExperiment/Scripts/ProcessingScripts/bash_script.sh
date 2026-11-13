#!/bin/bash

for i in {0..7}; do
    sbatch "./scripts/JammingExperiment/Scripts/ProcessingScripts/slurm_scripts/run_sensitivity_${i}.sbatch"
done