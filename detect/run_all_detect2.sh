#!/bin/bash

# =======================
# 1. SYNTH MODELS 列表
# =======================

SYNTH_MODEL="gpt-5-2025-08-07"
DETECT_MODEL="gpt-5-2025-08-07"

ROOT_DIR="/mnt/parallel_ssd/home/zdhs0006/ACL/data/ICML_30"
SYNTH_GLOB="paper_synth_*.json"
JOBS=10                # 并行任务数
OVERWRITE=""   # 或者改为 "" 禁止覆盖
OVERWRITE_ZERO="--overwrite_zero"   # 或者改为 "" 禁止覆盖

# python 4_1_mas_error_detection.py \
#     --root_dir "$ROOT_DIR" \
#     --synth_glob "$SYNTH_GLOB" \
#     --detect_mode fast \
#     --detect_model "$DETECT_MODEL" \
#     --synth_model "$SYNTH_MODEL" \
#     --jobs $JOBS \
#     $OVERWRITE \
#     $OVERWRITE_ZERO \
#     --disable_memory_build \
#     --disable_section_review \
#     --disable_per_task \

# python 4_1_mas_error_detection.py \
#     --root_dir "$ROOT_DIR" \
#     --synth_glob "$SYNTH_GLOB" \
#     --detect_mode standard \
#     --detect_model "$DETECT_MODEL" \
#     --synth_model "$SYNTH_MODEL" \
#     --jobs $JOBS \
#     $OVERWRITE \
#     $OVERWRITE_ZERO \
#     --disable_global_review \
#     --disable_per_task \

python 4_1_mas_error_detection.py \
    --root_dir "$ROOT_DIR" \
    --synth_glob "$SYNTH_GLOB" \
    --detect_mode standard_wo_memory \
    --detect_model "$DETECT_MODEL" \
    --synth_model "$SYNTH_MODEL" \
    --jobs $JOBS \
    $OVERWRITE \
    $OVERWRITE_ZERO \
    --disable_global_review \
    --disable_memory_build \
    --disable_per_task \

python 4_1_mas_error_detection.py \
    --root_dir "$ROOT_DIR" \
    --synth_glob "$SYNTH_GLOB" \
    --detect_mode deep \
    --detect_model "$DETECT_MODEL" \
    --synth_model "$SYNTH_MODEL" \
    --jobs $JOBS \
    $OVERWRITE \
    $OVERWRITE_ZERO \
    --disable_global_review \
    --disable_section_review \