#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5,6,746
export VLLM_USE_MODELSCOPE=False
export VLLM_LOGGING_LEVEL=ERROR


python -m vllm.entrypoints.openai.api_server \
  --model /Reward_Model/Qwen3-80B/ \
  --port 8001 \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 8192\
  --max-model-len 8192 \
  --disable-log-requests \
  --enforce-eager \
  --enable-prefix-caching \
  --max-num-seqs 64