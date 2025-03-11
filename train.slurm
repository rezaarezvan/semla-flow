#!/bin/bash
#SBATCH -A NAISS2024-22-1625
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 48:00:00
#SBATCH --job-name=semlaflow_qm9
#SBATCH --output=semlaflow_qm9_%j.log
#SBATCH --error=semlaflow_qm9_%j.err
set -e
echo "Running on $(hostname)"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Load necessary modules
module purge
module load Python/3.11

# Create and activate virtual environment if it doesn't exist
VENV_DIR="$HOME/semla"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate

# Install dependencies if needed
if [ ! -d "$VENV_DIR/lib/python3.11/site-packages/rdkit" ]; then
    echo "Installing dependencies..."
    pip install --no-cache-dir --upgrade pip
    pip install numpy==1.26.2 pandas==2.2.2 scipy==1.11.4 rdkit
    pip install lightning torchmetrics openbabel-wheel typing_extensions wandb tqdm
    pip install torch torchvision torchaudio
    pip install matplotlib jupyter ipykernel py3Dmol
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)
# Replace with API key
export WANDB_API_KEY=wandb_api_key

# Check dataset availability
DATA_PATH="/data/qm9/smol"
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Dataset not found at $DATA_PATH"
    exit 1
fi

python -m semlaflow.train \
    --data_path $DATA_PATH \
    --dataset qm9 \
    --d_model 384 \
    --n_layers 12 \
    --d_message 128 \
    --d_edge 128 \
    --n_coord_sets 64 \
    --n_attn_heads 32 \
    --d_message_hidden 128 \
    --coord_norm "length" \
    --size_emb 64 \
    --max_atoms 32 \
    --epochs 300 \
    --lr 0.0003 \
    --batch_cost 4096 \
    --acc_batches 1 \
    --gradient_clip_val 1.0 \
    --type_loss_weight 0.2 \
    --bond_loss_weight 0.5 \
    --warm_up_steps 2000 \
    --categorical_strategy "uniform-sample" \
    --lr_schedule "constant" \
    --bucket_cost_scale "linear" \
    --val_check_epochs 10 \
    --n_validation_mols 2000 \
    --integration_steps 100 \
    --coord_noise_std_dev 0.2 \
    --optimal_transport "equivariant" \
    --self_condition

echo "End time: $(date)"
