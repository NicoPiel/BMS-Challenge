#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --tasks=16
#SBATCH --mem=180G
#SBATCH --time=2:00:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

# load modules
module load python3 gcc/7.3.0 cuda

# start the model
python 2_Model.py && python 3_Inference.py