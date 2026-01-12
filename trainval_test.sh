#!/bin/bash
#SBATCH-J CrowdSatDataset_test
#SBATCH-p intel
#SBATCH-N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Output logs
#SBATCH -o output_log/%j.out
#SBATCH -e output_log/%j.err

source /share/home/u18042/miniconda3/etc/profile.d/conda.sh
conda activate lytorch
srun python trainval.py