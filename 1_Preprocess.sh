#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --tasks=16
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

# load modules
module load python3 gcc/7.3.0 cuda

# start the model
ipython ./1_Preprocess.py
