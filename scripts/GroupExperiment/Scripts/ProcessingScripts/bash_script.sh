#!/bin/bash

for i in {0..7}; do
    sbatch "./scripts/GroupExperiment/Scripts/ProcessingScripts/slurm_scripts/run_sensitivity_${i}.sbatch"
done