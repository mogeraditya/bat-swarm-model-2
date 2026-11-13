#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # only start 1 task via srun because Python multiprocessing starts more tasks internally
#SBATCH --cpus-per-task=128  
#SBATCH --time 24:00:00

#SBATCH --output=/viper/u1/amoger/bat-swarm-model/slurm_logs/jammingexperiment_14_10_%j.out
#SBATCH --error=/viper/u1/amoger/bat-swarm-model/slurm_logs/jammingexperiment_14_10_%j.err

#SBATCH --job-name=dynamic_model
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aditya.moger@uni-konstanz.de


# Recommended: prevent nested threading inside each worker process
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1


module load anaconda/3/2023.03

conda init bash
source ~/.bashrc
conda activate /u/amoger/conda-envs/dynamic_model

# Use srun inside allocations (good Slurm hygiene)
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  python -u \
  scripts/JammingExperiment/Scripts/ProcessingScripts/run_sweep.py \
  -m 