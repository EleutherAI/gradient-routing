#!/bin/bash
#SBATCH --job-name=gr-concept-pt
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/home/a6a/lucia.a6a/gradient-routing/runs/tinystories-concept-posttrain-%j.out

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate gradient-routing

module load PrgEnv-cray
module load cuda/12.6

export WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/home/a6a/lucia.a6a/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/home/a6a/lucia.a6a/.cache/torch_inductor

cd /home/a6a/lucia.a6a/gradient-routing

echo "Starting per-token (concept) post-training ERA gradient routing experiment"
echo "Time: $(date)"
nvidia-smi

python projects/tinystories/tinystories_era_concept_posttrain.py

echo "Done: $(date)"
