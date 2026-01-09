#!/bin/bash
# Run train+prune.py for all 5 DINOv3 model configurations

set -e  # Exit on error

CONFIG_FILE="config.ini"

# Function to update config.ini
update_config() {
    local encoder=$1
    local model_name=$2
    local lr=$3

    echo "Configuring: encoder=$encoder, model=$model_name, lr=$lr"

    # Update MODEL_ENCODER
    sed -i "s/^MODEL_ENCODER = .*/MODEL_ENCODER = $encoder/" "$CONFIG_FILE"

    # Update MODEL_NAME
    sed -i "s/^MODEL_NAME = .*/MODEL_NAME = $model_name/" "$CONFIG_FILE"

    # Update LR
    sed -i "s/^LR = .*/LR = $lr/" "$CONFIG_FILE"
}

# ConvNeXt variants (LR = 0.005)
CONVNEXT_MODELS=(
    "convnext_tiny.dinov3_lvd1689m"
    "convnext_small.dinov3_lvd1689m"
    "convnext_base.dinov3_lvd1689m"
)

# ViT variants (LR = 0.002)
VIT_MODELS=(
    "vit_small_patch16_dinov3.lvd1689m"
    "vit_base_patch16_dinov3.lvd1689m"
)

echo "=========================================="
echo "Starting training for all 5 configurations"
echo "=========================================="

# Run ConvNeXt models
for model in "${CONVNEXT_MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training: $model"
    echo "=========================================="
    update_config "convnext" "$model" "0.005"
    python train+prune.py
done

# Run ViT models
for model in "${VIT_MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training: $model"
    echo "=========================================="
    update_config "vit" "$model" "0.002"
    python train+prune.py
done

echo ""
echo "=========================================="
echo "All 5 configurations completed!"
echo "=========================================="
