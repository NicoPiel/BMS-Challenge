#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --tasks=4
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

# load modules
module load python3 gcc/7.3.0 cuda

# start the model
ipython 2_Model.py -- -n 1 -g 4 -nr 0 ---epochs 9