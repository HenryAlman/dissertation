#!/bin/bash

#SBATCH --job-name=henry_diss
#SBATCH --output=slurmlogs/job_%A_%a.out
#SBATCH --error=slurmlogs/job_%A_%a.err
#SBATCH --partition=standard
#SBATCH --time=28:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus_per_task=1
#SBATCH --mem=2G

#SBATCH --array=1-60

module purge
cd /users/40795510/sharedscratch/dissertation
module load apps/anaconda3/2024.10
source activate ./test_venv

ARGS=$(sed -n "{$SLURM_ARRAY_TASK_ID}p" params.text)

echo "Starting task ${$SLURM_ARRAY_TASK_ID}"
echo "Args: ${ARGS}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

python3 PythonDissertation/mapelites_mujoco_test_cpg_3.py $ARGS

conda deactivate
