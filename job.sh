#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --tasks=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

# load modules
module load cuda/10.2

# start preprocessing
ipython ./1_Preprocess.py
# compile and train model
ipython ./2_Model.py
# start inference
ipython ./3_Inference.py