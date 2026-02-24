#!/bin/bash
#SBATCH --job-name=vir-era-dry
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/a6a/lucia.a6a/gradient-routing/runs/virology-era-dryrun-%j.out

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate gradient-routing

module load PrgEnv-cray
module load cuda/12.6

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/home/a6a/lucia.a6a/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/home/a6a/lucia.a6a/.cache/torch_inductor
export HF_HUB_OFFLINE=1

cd /home/a6a/lucia.a6a/gradient-routing

echo "=== Virology ERA Dry Run ==="
echo "Time: $(date)"
echo "Node: $(hostname)"
nvidia-smi

python projects/virology_era/virology_era.py dry_run

echo "Done: $(date)"
