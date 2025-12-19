#!/bin/bash


# =======================
# 1. SYNTH MODELS 列表
# =======================
SYNTH_MODELS=(
    "gpt-5-2025-08-07"
    "o4-mini"
    "gemini-2.5-pro"
    "claude-sonnet-4-5-20250929"
    "grok-4"
    "qwen3-vl-235b-a22b-instruct"
    "doubao-seed-1-6-251015"
    "glm-4.6"
)

# =======================
# 2. DETECT MODELS 列表
# =======================
DETECT_MODELS=(
    "gpt-5-2025-08-07"
    "o4-mini"
    "gemini-2.5-pro"
    "claude-sonnet-4-5-20250929"
    "grok-4"
    "doubao-seed-1-6-251015"
    "glm-4.6"
    ##### TEXT #####
    "deepseek-v3.1"
    "kimi-k2-250905"
    "qwen3-235b-a22b-instruct-2507"
)

EVAL_MODES=(
    "fast"
    # "standard"
    # "deep"
    # "deep_standard"
    # "standard_wo_memory"
)


# =======================
# 3. 共同参数（按需修改）
# =======================
ROOT_DIR="/mnt/parallel_ssd/home/zdhs0006/ACL/data/NeurIPS_35"
JOBS=10                # 并行任务数
OVERWRITE=""   # 或者改为 "" 禁止覆盖
EVAL_MODEL="gpt-5.1"

# =======================
# 4. 双重循环执行检测
# =======================

for synth in "${SYNTH_MODELS[@]}"; do
    for detect in "${DETECT_MODELS[@]}"; do
        for mode in "${EVAL_MODES[@]}"; do
            echo "======================================================="
            echo " RUNNING DETECT_MODEL = $detect   SYNTH_MODEL = $synth "
            echo "======================================================="

            python 5_eval_detection.py \
                --root_dir "$ROOT_DIR" \
                --eval_mode "$mode" \
                --detect_model "$detect" \
                --synth_model "$synth" \
                --eval_model  "$EVAL_MODEL" \
                -j $JOBS \
                $OVERWRITE

            echo ""
            echo ">>> FINISHED: DETECT = $detect   SYNTH = $synth"
            echo ""
        done
    done
done

echo "======================"
echo "   ALL RUNS DONE!     "
echo "======================"
