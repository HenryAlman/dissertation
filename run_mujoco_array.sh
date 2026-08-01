#!/bin/bash

#SBATCH --job-name=bopelites
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
module load python/3.06
source my_env/bin/activate

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

deactivate