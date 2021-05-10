#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --tasks=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

# load modules
module load python3 gcc/7.3.0 cuda

# start the model
python -m torch.distributed.launch --nproc_per_node=4 2_Model.py && python -m torch.distributed.launch --nproc_per_node=4 3_Inference.py