#!/bin/bash
#SBATCH --tasks=16
#SBATCH --mem=12G

# load modules
module load cuda/10.2

# start preprocessing
ipython ./1_Preprocess.py
# compile and train model
ipython ./2_Model.py
# start inference
ipython ./3_Inference.py