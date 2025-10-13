#!/bin/bash
################################################################################
# Comprehensive Transfer Attack Experiments Script
#
# This script runs a systematic evaluation of multiple transfer-based attacks
# across different source models to measure their transferability.
#
# Experiment Configuration:
# - Source Models: Inc-v3, Inc-v4, IncRes-v2, Res-152
# - Target Models (evaluated): Inc-v3, Inc-v4, IncRes-v2, Res-50, Res-101, Res-152
# - Attack Methods: MI-FGSM, DI-MI-FGSM, TI-MI-FGSM, SI-MI-FGSM, VMI, ANDA, MUMODIG
#
# Note: RDI-FIM is not available in the current TransferAttack framework.
#       If you need this method, please verify the correct method name or
#       implement it as a new attack class.
################################################################################

# GPU configuration
GPU_ID=1

# Dataset paths
INPUT_DIR="./data"
BASE_OUTPUT_DIR="./results"

# Attack hyperparameters (standard settings for fair comparison)
EPSILON=0.0627     # 16/255 perturbation budget
ALPHA=0.00627      # 1.6/255 step size
EPOCH=10           # number of iterations
BATCHSIZE=1       # default batch size

# Source models for generating adversarial examples
# Model name mapping:
# - inception_v3: Inc-v3 (Inception v3) - timm standard weights
# - inception_v4: Inc-v4 (Inception v4) - timm standard weights
# - inception_resnet_v2: IncRes-v2 (Inception-ResNet v2) - timm standard weights
# - resnet152: Res-152 (ResNet-152) - timm standard weights
SOURCE_MODELS=(
    "inception_v3"
    "inception_v4"
    "inception_resnet_v2"
    "resnet152"
)

# Attack methods to evaluate
# Available attacks with their characteristics:
# 1. mifgsm: MI-FGSM (Momentum Iterative FGSM)
# 2. dim: DI-MI-FGSM (Diverse Input + MI-FGSM)
# 3. tim: TI-MI-FGSM (Translation-Invariant + MI-FGSM)
# 4. sim: SI-MI-FGSM (Scale-Invariant + MI-FGSM)
# 5. vmifgsm: VMI-FGSM (Variance-tuned MI-FGSM)
# 6. anda: ANDA (Asymptotically Normal Distribution Attack) - requires batchsize=1
# 7. mumodig: MUMODIG (Multi-Momentum Diverse Integrated Gradients)
#
# Note: RDI-FIM is NOT available in TransferAttack framework
ATTACK_METHODS=(
   "admix"
)

# Special attacks that require batchsize=1 for proper functionality
SPECIAL_BATCHSIZE_ATTACKS=(
    "anda"
)

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Log file for tracking experiment progress
LOG_FILE="${BASE_OUTPUT_DIR}/experiment_log.txt"
echo "=========================================" | tee -a "$LOG_FILE"
echo "Comprehensive Transfer Attack Experiments" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

# Main experiment loop: iterate over all source models and attack methods
for source_model in "${SOURCE_MODELS[@]}"; do
    for attack in "${ATTACK_METHODS[@]}"; do
        # Determine output directory for this experiment
        output_dir="${BASE_OUTPUT_DIR}/${attack}/${source_model}"

        # Check if this attack requires special batch size
        current_batchsize=$BATCHSIZE
        for special_attack in "${SPECIAL_BATCHSIZE_ATTACKS[@]}"; do
            if [ "$attack" = "$special_attack" ]; then
                current_batchsize=1
                break
            fi
        done

        echo "" | tee -a "$LOG_FILE"
        echo "=========================================" | tee -a "$LOG_FILE"
        echo "Experiment: ${attack} + ${source_model}" | tee -a "$LOG_FILE"
        echo "=========================================" | tee -a "$LOG_FILE"
        echo "Source Model: ${source_model}" | tee -a "$LOG_FILE"
        echo "Attack Method: ${attack}" | tee -a "$LOG_FILE"
        echo "Batch Size: ${current_batchsize}" | tee -a "$LOG_FILE"
        echo "Output Directory: ${output_dir}" | tee -a "$LOG_FILE"
        echo "Started at: $(date)" | tee -a "$LOG_FILE"
        echo "-----------------------------------------" | tee -a "$LOG_FILE"

        # Generate adversarial examples (evaluation disabled)
        echo "Generating adversarial examples..." | tee -a "$LOG_FILE"
        python main.py \
            --input_dir "$INPUT_DIR" \
            --output_dir "$output_dir" \
            --attack "$attack" \
            --model "$source_model" \
            --batchsize $current_batchsize \
            --eps $EPSILON \
            --alpha $ALPHA \
            --epoch $EPOCH \
            --GPU_ID $GPU_ID

        # Check if adversarial example generation succeeded
        if [ $? -eq 0 ]; then
            echo "[✓] Successfully generated adversarial examples" | tee -a "$LOG_FILE"
            echo "[INFO] Skipping evaluation (generate-only mode)" | tee -a "$LOG_FILE"
        else
            echo "[✗] Error generating adversarial examples for ${attack} on ${source_model}" | tee -a "$LOG_FILE"
        fi

        echo "Finished at: $(date)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done
done

# Experiment completion summary
echo "=========================================" | tee -a "$LOG_FILE"
echo "All experiments completed!" | tee -a "$LOG_FILE"
echo "Completed at: $(date)" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "Results saved to: ${BASE_OUTPUT_DIR}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Experiment Summary:" | tee -a "$LOG_FILE"
echo "- Source Models: ${#SOURCE_MODELS[@]}" | tee -a "$LOG_FILE"
echo "- Attack Methods: ${#ATTACK_METHODS[@]}" | tee -a "$LOG_FILE"
echo "- Total Experiments: $((${#SOURCE_MODELS[@]} * ${#ATTACK_METHODS[@]}))" | tee -a "$LOG_FILE"
echo "- Mode: Generate-Only (evaluation disabled)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Generated adversarial examples:" | tee -a "$LOG_FILE"
echo "  Location: ${BASE_OUTPUT_DIR}/<attack>/<model>/" | tee -a "$LOG_FILE"
echo "  Example: ${BASE_OUTPUT_DIR}/mifgsm/inception_v3/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To evaluate the generated samples, run:" | tee -a "$LOG_FILE"
echo "  bash evaluate_all_samples.sh" | tee -a "$LOG_FILE"
echo "Or evaluate individually:" | tee -a "$LOG_FILE"
echo "  python main.py --input_dir ./data --output_dir ${BASE_OUTPUT_DIR}/<attack>/<model> --eval --GPU_ID 0" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
