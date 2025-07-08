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


cd /workspace/mbekbergenova/spectra_transformer

# Array of seeds to test
seeds=(1 42 123 456 789)

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "Starting training with seed: $seed"

    # Run training with current seed
    srun python Scripts/train_on_cluster_devices.py \
        --file_paths Scripts/file_paths.txt \
        --log_dir "./cluster_logs/seed_${seed}" \
        --num_workers 8 \
        --seed $seed

    # Check if the previous command was successful
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for seed: $seed"
    else
        echo "Training failed for seed: $seed"
        exit 1
    fi

    # Optional: Add a small delay between runs
    sleep 10
done

echo "All training runs completed!"


