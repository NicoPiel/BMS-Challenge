#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --tasks=1
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

# load modules
module load python3 gcc/7.3.0 cuda

# start the model
ipython 3_Inference.py