#!/bin/bash
################################################################################
# Evaluation Script for Generated Adversarial Examples
#
# This script evaluates all previously generated adversarial examples
# on the target models to measure attack success rate (ASR).
################################################################################

# GPU configuration
GPU_ID=0

# Dataset paths
INPUT_DIR="./data"
BASE_OUTPUT_DIR="./results"
BATCHSIZE=16

# Log file
LOG_FILE="${BASE_OUTPUT_DIR}/evaluation_log.txt"

echo "=========================================" | tee "$LOG_FILE"
echo "Evaluating All Adversarial Examples" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Find all generated adversarial example directories
# Format: results/<attack>/<model>/
eval_dirs=$(find "$BASE_OUTPUT_DIR" -type d -path "*/results/*/*" | sort)

if [ -z "$eval_dirs" ]; then
    echo "No adversarial examples found in ${BASE_OUTPUT_DIR}" | tee -a "$LOG_FILE"
    echo "Please run generation first: bash run_comprehensive_experiments.sh" | tee -a "$LOG_FILE"
    exit 1
fi

# Count total directories
total_dirs=$(echo "$eval_dirs" | wc -l)
current=0

echo "Found ${total_dirs} directories to evaluate" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Evaluate each directory
while IFS= read -r output_dir; do
    current=$((current + 1))

    # Extract attack and model from path
    attack=$(basename $(dirname "$output_dir"))
    model=$(basename "$output_dir")

    echo "=========================================" | tee -a "$LOG_FILE"
    echo "Evaluation ${current}/${total_dirs}" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"
    echo "Attack: ${attack}" | tee -a "$LOG_FILE"
    echo "Source Model: ${model}" | tee -a "$LOG_FILE"
    echo "Directory: ${output_dir}" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "-----------------------------------------" | tee -a "$LOG_FILE"

    # Check if adversarial examples exist
    num_images=$(find "$output_dir" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)

    if [ "$num_images" -eq 0 ]; then
        echo "[⚠] No images found, skipping..." | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        continue
    fi

    echo "Found ${num_images} adversarial images" | tee -a "$LOG_FILE"
    echo "Evaluating on all target models..." | tee -a "$LOG_FILE"

    # Run evaluation
    python main.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$output_dir" \
        --eval \
        --batchsize $BATCHSIZE \
        --GPU_ID $GPU_ID

    if [ $? -eq 0 ]; then
        echo "[✓] Successfully evaluated ${attack}/${model}" | tee -a "$LOG_FILE"
    else
        echo "[✗] Error evaluating ${attack}/${model}" | tee -a "$LOG_FILE"
    fi

    echo "Finished at: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done <<< "$eval_dirs"

# Summary
echo "=========================================" | tee -a "$LOG_FILE"
echo "Evaluation Complete!" | tee -a "$LOG_FILE"
echo "Completed at: $(date)" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "Total evaluations: ${total_dirs}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results:" | tee -a "$LOG_FILE"
echo "  - Detailed log: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "  - ASR table: results_eval.txt" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To view results:" | tee -a "$LOG_FILE"
echo "  cat results_eval.txt" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
