#!/bin/bash

# =======================
# 1. SYNTH MODELS List
# =======================
SYNTH_MODELS=(
    "gpt-5-2025-08-07"
    # "o4-mini"
    # "gemini-2.5-pro"
    # "claude-sonnet-4-5-20250929"
    # "grok-4"
    # "qwen3-vl-235b-a22b-instruct"
    # "doubao-seed-1-6-251015"
    # "glm-4.6"
)

# =======================
# 2. DETECT MODELS List
# =======================
DETECT_MODELS=(
    "gpt-5-2025-08-07"
    # "o4-mini"
    # "gemini-2.5-pro"
    # "claude-sonnet-4-5-20250929"
    # "grok-4"
    # "doubao-seed-1-6-251015"
    # "glm-4.6"
    ##### TEXT #####
    # "deepseek-v3.1"
    # "kimi-k2-250905"
    # "qwen3-235b-a22b-instruct-2507"
)

# =======================
# 3. EVAL MODES List (should match detect modes: fast, standard, deep)
# =======================
EVAL_MODES=(
    "fast"
    # "standard"
    # "deep"
)

# =======================
# 4. Common Parameters (modify as needed)
# =======================
ROOT_DIR="/path/to/papers"
JOBS=10                # Number of concurrent evaluation tasks
OVERWRITE=""   # Set to "" to disable overwrite
EVAL_MODEL="gpt-5.1"

# =======================
# 5. Triple loop execution (synth × detect × mode)
# =======================

for synth in "${SYNTH_MODELS[@]}"; do
    for detect in "${DETECT_MODELS[@]}"; do
        for mode in "${EVAL_MODES[@]}"; do
            echo "======================================================="
            echo " RUNNING MODE = $mode   DETECT_MODEL = $detect   SYNTH_MODEL = $synth "
            echo "======================================================="

            python eval_detection.py \
                --root_dir "$ROOT_DIR" \
                --eval_mode "$mode" \
                --detect_model "$detect" \
                --synth_model "$synth" \
                --eval_model  "$EVAL_MODEL" \
                -j $JOBS \
                $OVERWRITE

            echo ""
            echo ">>> FINISHED: MODE = $mode   DETECT = $detect   SYNTH = $synth"
            echo ""
        done
    done
done

echo "======================"
echo "   ALL RUNS DONE!     "
echo "======================"
