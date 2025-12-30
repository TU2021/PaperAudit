# RL Training Project

This project implements reinforcement learning (RL) training for large language models using the VERL framework with Group Relative Policy Optimization (GRPO) algorithm. The training focuses on academic paper error detection tasks, using a custom reward function that leverages Qwen3-80B as a judge model.

## Project Overview

This project supports RL training for multiple language models:
- **Qwen3-8B**: Qwen 8B model
- **Qwen3-14B**: Qwen 14B model
- **Llama-3.2-3B-Instruct**: Meta Llama 3.2 3B instruction model

The training uses GRPO algorithm to optimize model performance on scientific error detection tasks, where the model must identify errors, methodological flaws, or academic integrity issues in paper text.

## Environment Setup

### Using Conda Environment

The project provides an `environment.yml` file. You can create the environment using:

```bash
conda env create -f environment.yml
conda activate rl
```

### Using pip Installation

If you already have a Python environment, install dependencies directly:

```bash
pip install -r requirements.txt
```

## Key Dependencies

- **VERL**: Volcengine Reinforcement Learning framework for LLM training
- **vLLM**: High-throughput LLM inference and serving engine
- **PyTorch**: Deep learning framework
- **Ray**: Distributed computing framework
- **Hydra**: Configuration management framework
- **OpenAI API Client**: For interacting with the judge model API

## Project Structure

```
rl_training/
├── readme.md                    # This file
├── run_grpo_train.sh            # Main GRPO training script
├── start_qwen80b_vllm.sh        # Script to start Qwen3-80B judge model server
├── reward_function.py           # Custom reward function implementation
├── requirements.txt              # Python dependencies
├── environment.yml              # Conda environment configuration
├── config/
│   └── grpo_config.yaml         # GRPO training configuration
├── reward_model/
│   └── Qwen3-80B/              # Qwen3-80B judge model directory
├── qwen3_8b_rl/                # Qwen3-8B RL training outputs
├── qwen3_14b_rl/               # Qwen3-14B RL training outputs
└── llama3.2_3b_rl/             # Llama-3.2-3B RL training outputs
```

## Usage

### 1. Start the Judge Model Server

Before training, you need to start the Qwen3-80B judge model server using vLLM:

```bash
bash start_qwen80b_vllm.sh
```

This will start an OpenAI-compatible API server on port 8001. The server uses:
- 4 GPUs with tensor parallelism
- bfloat16 precision
- 85% GPU memory utilization
- Maximum sequence length of 8192 tokens

### 2. Configure Training Parameters

Edit `run_grpo_train.sh` to configure:

- **Data paths**: Set `TRAIN_DATA` and `VAL_DATA` to your training/validation parquet files
- **Model path**: Update `actor_rollout_ref.model.path` to point to your base model
- **Output directory**: Modify `ROOT_EXP_DIR` for experiment outputs
- **Reward function**: Ensure `custom_reward_function.path` points to the correct reward function file
- **GPU configuration**: Adjust `CUDA_VISIBLE_DEVICES` and `trainer.n_gpus_per_node` based on available GPUs

### 3. Configure Reward Function

Edit `reward_function.py` to customize:

- **Judge model API**: Update `JUDGE_BASE_URL` to match your judge model server
- **Reward weights**: Adjust `PRECISION_WEIGHT` and `CONCISENESS_WEIGHT` (default: 0.6 and 0.4)
- **Penalty values**: Configure penalty scores for invalid outputs
- **Evaluation prompt**: Modify `REWARD_PROMPT` to change evaluation criteria

### 4. Run Training

Execute the training script:

```bash
bash run_grpo_train.sh
```

The script will:
- Create experiment directories with timestamps
- Launch distributed training using Ray
- Generate rollouts using vLLM
- Compute rewards using the custom reward function
- Update the policy using GRPO algorithm
- Log training metrics to console and SwanLab

## Training Configuration

### GRPO Algorithm Parameters

Key parameters in `config/grpo_config.yaml`:

- `algorithm.adv_estimator: grpo`: Use GRPO advantage estimator
- `algorithm.norm_adv_by_std_in_grpo: true`: Normalize advantages by standard deviation
- `algorithm.use_kl_in_reward: false`: Whether to use KL divergence in reward
- `algorithm.grpo.clip_adv: true`: Clip advantages
- `algorithm.grpo.clip_adv_value: 5.0`: Advantage clipping value
- `algorithm.grpo.gamma: 1.0`: Discount factor
- `algorithm.grpo.lam: 0.95`: GAE lambda parameter

### Training Hyperparameters

Main hyperparameters (configurable in `run_grpo_train.sh`):

- `data.train_batch_size: 64`: Total training batch size
- `data.micro_batch_size_per_gpu: 8`: Micro batch size per GPU
- `actor_rollout_ref.rollout.n: 2`: Number of samples per prompt
- `actor_rollout_ref.actor.ppo_mini_batch_size: 64`: PPO mini-batch size
- `trainer.total_epochs: 3`: Number of training epochs
- `trainer.n_gpus_per_node: 8`: Number of GPUs per node
- `trainer.save_freq: 100`: Checkpoint saving frequency
- `trainer.test_freq: 50`: Validation frequency

### Rollout Configuration

- `actor_rollout_ref.rollout.max_model_len: 12288`: Maximum model context length
- `actor_rollout_ref.rollout.max_num_batched_tokens: 8192`: Maximum batched tokens
- `actor_rollout_ref.rollout.gpu_memory_utilization: 0.6`: GPU memory utilization for vLLM
- `data.max_prompt_length: 4096`: Maximum prompt length
- `data.max_response_length: 2048`: Maximum response length

## Reward Function

The custom reward function (`reward_function.py`) evaluates model outputs based on:

### Evaluation Criteria

1. **Precision (weight: 0.6)**:
   - Semantic match with ground truth errors
   - Penalties for false positives (-0.2 per error)
   - Penalties for duplicate errors (-0.4 per duplicate)
   - No penalty for missing errors (prioritizes accuracy over completeness)

2. **Conciseness (weight: 0.4)**:
   - No redundant information: 1.0 points
   - Minor redundancy: 0.7 points
   - Excessive redundancy: 0.3 points

### Reward Computation

The final reward is computed as:
```
total_reward = PRECISION_WEIGHT * precision + CONCISENESS_WEIGHT * conciseness
```

The reward function includes:
- **Caching**: MD5-based caching to avoid redundant judge model calls
- **Error handling**: Graceful handling of API failures and invalid outputs
- **Output cleaning**: Removes `<think>` tags and other artifacts

## Monitoring

Training progress is monitored through:

- **Console logs**: Real-time training metrics and debug information
- **SwanLab**: Experiment tracking and visualization (configured via `trainer.logger`)
- **Checkpoints**: Model checkpoints saved at specified intervals
- **Validation**: Periodic validation runs with generation logging

## Notes

1. **Judge Model Server**: Ensure the Qwen3-80B judge model server is running before starting training
2. **GPU Requirements**: Training requires multiple GPUs (default: 8 GPUs). Adjust based on model size and available resources
3. **Data Format**: Training data should be in parquet format with `prompt` and `ground_truth` columns
4. **VERL Framework**: This project uses VERL framework. Ensure the VERL repository is cloned and the path is correctly set in `PYTHONPATH`
5. **Network Configuration**: The reward function makes HTTP requests to the judge model API. Ensure network connectivity and proper firewall settings
6. **Memory Management**: vLLM uses significant GPU memory. Adjust `gpu_memory_utilization` if encountering OOM errors

## Troubleshooting

### Common Issues

1. **Judge model API connection failed**:
   - Verify the judge model server is running: `curl http://10.1.100.72:8001/v1/models`
   - Check network connectivity and firewall settings
   - Verify `JUDGE_BASE_URL` in `reward_function.py`

2. **Out of memory errors**:
   - Reduce `gpu_memory_utilization` in rollout configuration
   - Decrease `max_num_batched_tokens`
   - Reduce batch sizes

3. **Ray initialization errors**:
   - Check Ray installation and version compatibility
   - Verify NCCL configuration for multi-GPU training
   - Ensure proper network setup for distributed training

## Related Resources

- [VERL Framework](https://github.com/volcengine/verl)
- [vLLM Documentation](https://docs.vllm.ai/)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Ray Documentation](https://docs.ray.io/)

