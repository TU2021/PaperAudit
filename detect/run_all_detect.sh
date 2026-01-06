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
# 3. DETECT MODES List
# =======================
DETECT_MODES=(
    "fast"
    # "standard"
    # "deep"
)

# =======================
# 4. Common Parameters (modify as needed)
# =======================
ROOT_DIR="/path/to/papers"
JOBS=10                # Thread pool size
OVERWRITE=""   # Set to "" to disable overwrite
OVERWRITE_ZERO="--overwrite_zero"   # Set to "" to disable overwrite_zero

# =======================
# 5. Triple loop execution (synth × detect × mode)
# =======================

for synth in "${SYNTH_MODELS[@]}"; do
    for detect in "${DETECT_MODELS[@]}"; do
        for mode in "${DETECT_MODES[@]}"; do

            echo "======================================================="
            echo " RUNNING MODE = $mode   DETECT_MODEL = $detect   SYNTH_MODEL = $synth "
            echo "======================================================="

            # Build command based on mode
            CMD="python mas_error_detection.py \
                --root_dir \"$ROOT_DIR\" \
                --detect_model \"$detect\" \
                --synth_model \"$synth\" \
                --jobs $JOBS \
                --detect_mode $mode"

            # Add mode-specific flags
            case "$mode" in
                "fast")
                    CMD="$CMD --enable-global-review"
                    ;;
                "standard")
                    CMD="$CMD --enable-memory-build --enable-section-review"
                    ;;
                "deep")
                    CMD="$CMD --enable-memory-build --enable-section-review --enable-per-task --enable-retriever --enable-memory-injection --enable-merge"
                    ;;
            esac

            # Add common flags
            CMD="$CMD $OVERWRITE $OVERWRITE_ZERO"

            # Execute command
            eval $CMD

            echo ""
            echo ">>> FINISHED: MODE = $mode   DETECT = $detect   SYNTH = $synth"
            echo ""
        done
    done
done

echo "======================"
echo "   ALL RUNS DONE!     "
echo "======================"
