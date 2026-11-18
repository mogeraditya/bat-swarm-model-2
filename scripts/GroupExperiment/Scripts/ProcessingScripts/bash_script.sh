#!/bin/bash

for i in {7..8}; do
    sbatch "./scripts/GroupExperiment/Scripts/ProcessingScripts/slurm_scripts/run_sensitivity_${i}.sbatch"
done