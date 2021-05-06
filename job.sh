#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --tasks=16
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

# load modules
module load python3 gcc/7.3.0 cuda

# start the model
ipython ./1_Preprocess.py && ipython ./2_Model.py && ipython ./3_Inference.py
