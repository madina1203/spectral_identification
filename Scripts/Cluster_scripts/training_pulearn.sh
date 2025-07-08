#!/bin/bash

#SBATCH --job-name=spectra_transformer__devices
#SBATCH --output=logs/training_%j.log
#SBATCH --error=logs/training_%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                         # Request 2 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --account=spectrometry



#SBATCH --mail-user=madina.bekbergenova@etu.univ-cotedazur.fr
#SBATCH --mail-type=BEGIN,END,FAIL

# Load necessary modules
module purge
module load cuda/12.8
module load miniconda/25.1.1

# Source conda initialization script
source /softs/miniconda3/etc/profile.d/conda.sh

# Skip conda activate, directly prepend env to PATH
export PATH="$HOME/.conda/envs/instrument_setting/bin:$PATH"
# Debug: confirm environment and torch
echo "Python path: $(which python)"
python -c "import sys; print('Python sys.path:', sys.path)"
python -c "import torch; print('Torch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
#Define the file path to h5 file

cd /workspace/mbekbergenova/spectra_transformer
# Run your training script
srun python Scripts/train_pulearn_gmm_v2.py\
    --file_paths Scripts/file_paths.txt \
    --log_dir ./cluster_logs \
    --num_workers 8




