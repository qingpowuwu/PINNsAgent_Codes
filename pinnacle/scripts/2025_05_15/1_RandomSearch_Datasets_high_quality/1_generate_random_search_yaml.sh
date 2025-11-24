#!/bin/bash

# Batch generate random search configurations
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle/scripts/2025_05_15/1_RandomSearch_Datasets_high_quality

# Set parameters directly, no external input needed
BATCH_COUNT=10   # Generate 10 batches
NEXP=30         # Generate 30 configurations per PDE

# Automatically get current date
CURRENT_DATE=$(date +%Y_%m_%d)  # Use underscore format

echo "Starting to generate $BATCH_COUNT batches, $NEXP configurations per PDE..."
echo "Current date: $CURRENT_DATE"

for ((i=1; i<=BATCH_COUNT; i++))
do
    BATCH_NAME="Batch_$i"
    echo "Generating $BATCH_NAME ..."
    
    # Configuration file save location
    OUT_DIR="./1_ICML_2025_configs/$BATCH_NAME"
    
    # Value of output_dir field in YAML
    YAML_OUTPUT_DIR="${CURRENT_DATE}/1_RandomSearch_Datasets_high_quality/1_ICML_2025_configs/$BATCH_NAME"
    
    # Generate 1D PDE configurations
    echo "  Generating 1D PDE configurations..."
    python 1_random_search-yaml-argsparser.py \
        --pde_type 1d \
        --device 0 \
        --nexp $NEXP \
        --out_dir $OUT_DIR \
        --yaml_output_dir $YAML_OUTPUT_DIR
    
    # Generate 2D PDE configurations
    echo "  Generating 2D PDE configurations..."
    python 1_random_search-yaml-argsparser.py \
        --pde_type 2d \
        --device 0 \
        --nexp $NEXP \
        --out_dir $OUT_DIR \
        --yaml_output_dir $YAML_OUTPUT_DIR
    
    # Generate 3D PDE configurations
    echo "  Generating 3D PDE configurations..."
    python 1_random_search-yaml-argsparser.py \
        --pde_type 3d \
        --device 0 \
        --nexp $NEXP \
        --out_dir $OUT_DIR \
        --yaml_output_dir $YAML_OUTPUT_DIR
    
    # Generate ND PDE configurations
    echo "  Generating ND PDE configurations..."
    python 1_random_search-yaml-argsparser.py \
        --pde_type nd \
        --device 0 \
        --nexp $NEXP \
        --out_dir $OUT_DIR \
        --yaml_output_dir $YAML_OUTPUT_DIR
    
    echo "$BATCH_NAME generation completed!"
    echo "------------------------"
done

echo "All $BATCH_COUNT batches generated successfully!"
echo "Directory structure:"
echo "./1_ICML_2025_configs/"
for ((i=1; i<=BATCH_COUNT; i++))
do
    echo "├── Batch_$i/"
    echo "│   ├── 1d/"
    echo "│   ├── 2d/"
    echo "│   ├── 3d/"
    echo "│   └── nd/"
done