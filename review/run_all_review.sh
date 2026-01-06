#!/bin/bash
#
# Batch runner for AuditAgent
# Executes audit review for multiple model combinations (synthetic × review models)
#
# Usage:
#   ./run_all_review.sh
#
# Configuration:
#   - Edit SYNTH_MODELS and REVIEW_MODELS arrays below
#   - Adjust ROOT_DIR, MODEL_TAG, JOBS, and feature flags as needed

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =======================
# Configuration
# =======================

# Synthetic models list (empty string = origin papers)
SYNTH_MODELS=(
    ""
    # "gpt-5-2025-08-07"
    # "o4-mini"
    # "gemini-2.5-pro"
    # "claude-sonnet-4-5-20250929"
    # "grok-4"
    # "qwen3-vl-235b-a22b-instruct"
    # "doubao-seed-1-6-251015"
    # "glm-4.6"
)

# Review models list
REVIEW_MODELS=(
    "gpt-5-2025-08-07"
    # "o4-mini"
    # "gemini-2.5-pro"
    # "claude-sonnet-4-5-20250929"
    # "grok-4"
    # "doubao-seed-1-6-251015"
    # "glm-4.6"
    # Text-only models:
    # "deepseek-v3.1"
    # "kimi-k2-250905"
    # "qwen3-235b-a22b-instruct-2507"
)

# Common parameters
ROOT_DIR="/mnt/parallel_ssd/home/zdhs0006/ACL/data_test"
MODEL_TAG="all"
JOBS=10

# Feature flags (set to flag string to enable, "" to disable)
OVERWRITE="--overwrite"              # Set to "--overwrite" to enable overwrite
NO_CHEAT=""               # Set to "--no_cheating_detection" to disable cheating detection
NO_MOTIVATION=""          # Set to "--no_motivation" to disable motivation evaluation
REUSE_CACHE=""            # Set to "--reuse_cache" to enable cache reuse

# Script to run
SCRIPT_NAME="run_audit_agent.py"

# =======================
# Validation
# =======================

if [[ ! -f "$SCRIPT_NAME" ]]; then
    echo "Error: Script '$SCRIPT_NAME' not found in current directory" >&2
    exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
    echo "Error: Root directory '$ROOT_DIR' does not exist" >&2
    exit 1
fi

# =======================
# Main execution
# =======================

TOTAL_RUNS=$((${#SYNTH_MODELS[@]} * ${#REVIEW_MODELS[@]}))
CURRENT_RUN=0
FAILED_RUNS=()

echo "======================================================="
echo " AuditAgent Batch Runner"
echo "======================================================="
echo "Root directory: $ROOT_DIR"
echo "Model tag: $MODEL_TAG"
echo "Jobs: $JOBS"
echo "Overwrite: ${OVERWRITE:+enabled}"
echo "Cheating detection: ${NO_CHEAT:+disabled}"
echo "Motivation evaluation: ${NO_MOTIVATION:+disabled}"
echo "Cache reuse: ${REUSE_CACHE:+enabled}"
echo "Total combinations: $TOTAL_RUNS"
echo "======================================================="
echo ""

for synth in "${SYNTH_MODELS[@]}"; do
    for review in "${REVIEW_MODELS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        synth_display="${synth:-origin}"
        
        echo "======================================================="
        echo "[$CURRENT_RUN/$TOTAL_RUNS] REVIEW_MODEL=$review  SYNTH_MODEL=$synth_display"
        echo "======================================================="
        
        if python "$SCRIPT_NAME" \
            --input_dir "$ROOT_DIR" \
            --model_tag "$MODEL_TAG" \
            --model "$review" \
            --synth_model "$synth" \
            --jobs "$JOBS" \
            $OVERWRITE \
            $NO_MOTIVATION \
            $NO_CHEAT \
            $REUSE_CACHE; then
            echo ""
            echo ">>> SUCCESS: REVIEW=$review  SYNTH=$synth_display"
            echo ""
        else
            echo ""
            echo ">>> FAILED: REVIEW=$review  SYNTH=$synth_display" >&2
            FAILED_RUNS+=("REVIEW=$review  SYNTH=$synth_display")
            echo ""
        fi
    done
done


echo "All runs completed successfully!"
exit 0
