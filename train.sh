#!/bin/bash
# ============================================================
# DIN-V2 Training Script
# Behavior-Type-Aware Deep Interest Network with Transformer
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo " DIN-V2: Enhanced Deep Interest Network Training Pipeline"
echo " Dataset: UserBehavior (Alibaba Taobao)"
echo " Time: $(date)"
echo "============================================================"

# Step 1: Generate/Download data
echo ""
echo "[Step 1/3] Preparing dataset..."
if [ ! -f "data/UserBehavior.csv" ]; then
    echo "Generating UserBehavior dataset (100M records)..."
    python3 data/download_data.py --num_records 100000000 --verify
else
    echo "Dataset already exists, skipping generation."
    ls -lh data/UserBehavior.csv
fi

# Step 2: Detect device
echo ""
echo "[Step 2/3] Detecting compute device..."
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="cuda"
    echo "GPU detected! Using CUDA."
    python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
else
    DEVICE="cpu"
    echo "No GPU detected, using CPU."
fi

# Step 3: Train DIN-V2
echo ""
echo "[Step 3/3] Training DIN-V2 model..."
echo "============================================================"

python3 src/train.py \
    --model v2 \
    --embed_dim 64 \
    --num_heads 4 \
    --num_transformer_layers 2 \
    --max_seq_len 50 \
    --dropout 0.1 \
    --epochs 3 \
    --batch_size 1024 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --patience 3 \
    --grad_clip 1.0 \
    --data_path data/UserBehavior.csv \
    --output_dir data/ \
    --num_workers 4 \
    --min_hist_len 5 \
    --device $DEVICE \
    --log_dir logs/ \
    --checkpoint_dir checkpoints/ \
    --log_interval 100 \
    --eval_interval 5000

echo ""
echo "============================================================"
echo " Training Complete! Check logs/ for training history."
echo "============================================================"
