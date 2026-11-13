#!/usr/bin/env bash
#SBATCH --job-name=dynamic_model
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # only start 1 task via srun because Python multiprocessing starts more tasks internally
#SBATCH --cpus-per-task=128  
#SBATCH --time=24:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=aditya.moger@uni-konstanz.de

#SBATCH --output=/viper/u1/amoger/bat-swarm-model/slurm_logs/jammingexperiment_14_10_%j.out
#SBATCH --error=/viper/u1/amoger/bat-swarm-model/slurm_logs/jammingexperiment_14_10_%j.err

set -euo pipefail

# Move to the repository. This ensures Hydra finds relative config paths.
cd /viper/u1/amoger/bat-swarm-model

# Create output directory if it does not already exist.
mkdir -p slurm_logs

# Activate Conda.
# Adjust this path if your Conda installation is elsewhere.
source /u/amoger/miniconda3/etc/profile.d/conda.sh
conda activate dynamic_model

# Avoid oversubscribing BLAS/OpenMP threads.
# With 128 Slurm CPUs, change these if your code is actually intended
# to parallelize through OpenMP / MKL.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

# Run your Hydra program.
# Add "-m param=value1,value2" for a Hydra multirun/sweep.

srun python ./scripts/JammingExperiment/Scripts/ProcessingScripts/subprocess_file_for_sweep.py $SLURM_CPUS_PER_TASK

echo "Finished: $(date)"