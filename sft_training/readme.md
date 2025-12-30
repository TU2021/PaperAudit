# SFT Training Project

This project is designed for supervised fine-tuning (SFT) of large language models using the LLaMA-Factory framework and LoRA technology for parameter-efficient fine-tuning.

## Project Overview

This project supports SFT training for multiple open-source large language models, including:
- **Qwen3-8B**: Qwen 8B model
- **Qwen3-14B**: Qwen 14B model
- **Llama-3.2-3B-Instruct**: Meta Llama 3.2 3B instruction model

## Environment Setup

### Using Conda Environment

The project provides an `environment.yml` file. You can create the environment using the following commands:

```bash
conda env create -f environment.yml
conda activate review
```

### Using pip Installation

If you already have a Python environment, you can directly install the dependencies:

```bash
pip install -r requirements.txt
```

## Key Dependencies

- **LLaMA-Factory**: Large model training framework
- **PyTorch**: Deep learning framework
- **DeepSpeed**: Distributed training acceleration
- **Transformers**: Hugging Face model library
- **PEFT**: Parameter-efficient fine-tuning library (for LoRA)

## File Descriptions

### Training Scripts

- `sft_train.sh`: Main training launch script containing complete training parameter configuration

### Configuration Files

- `ds_z3_config.json`: DeepSpeed ZeRO Stage 3 configuration file for optimizing GPU memory usage during large model training
- `merge_model.yaml`: Model merging configuration file for merging LoRA weights back into the base model

### Model Directories

After training, the following directory structure will be generated:
- `{model_name}/`: Original base model directory
- `{model_name}-lora/`: LoRA fine-tuned adapter weights directory
- `{model_name}-merged/`: Merged complete model directory

## Usage

### 1. Training Models

Modify the following parameters in `sft_train.sh`:

- `--model_name_or_path`: Base model path
- `--output_dir`: LoRA weights output directory
- `--dataset`: Training dataset name
- `--deepspeed`: DeepSpeed configuration file path

Then run:

```bash
bash sft_train.sh
```

### 2. Training Parameters

Main training parameters (configured in `sft_train.sh`):

- `--finetuning_type lora`: Use LoRA fine-tuning method
- `--lora_rank 64`: LoRA rank size
- `--lora_alpha 128`: LoRA alpha parameter
- `--lora_dropout 0.05`: LoRA dropout rate
- `--cutoff_len 20480`: Maximum sequence length
- `--per_device_train_batch_size 2`: Batch size per device
- `--gradient_accumulation_steps 8`: Gradient accumulation steps
- `--learning_rate 5e-5`: Learning rate
- `--num_train_epochs 3.0`: Number of training epochs
- `--bf16 True`: Use bfloat16 precision

### 3. Model Merging

After training, you can use LLaMA-Factory's export functionality to merge LoRA weights into the base model:

Modify the path configuration in `merge_model.yaml`, then run:

```bash
llamafactory-cli export merge_model.yaml
```

## DeepSpeed Configuration

The project uses DeepSpeed ZeRO Stage 3 for distributed training optimization:

- **ZeRO Stage 3**: Shards optimizer states, gradients, and parameters across multiple GPUs
- **BF16 Mixed Precision**: Automatically enables bfloat16 training
- **Communication Optimization**: Enables overlapping communication and contiguous gradients

## Training Monitoring

The training process supports the following monitoring methods:

- **TensorBoard**: Enable with `--report_to tensorboard`
- **Loss Curves**: Automatically plot loss curves with `--plot_loss True`
- **Log Output**: Output logs every 10 steps (`--logging_steps 10`)

## Notes

1. **GPU Memory Requirements**: Using DeepSpeed ZeRO Stage 3 can significantly reduce memory requirements, but sufficient GPU memory is still needed
2. **Data Paths**: Ensure training data paths are correctly configured
3. **Model Paths**: Replace the `xxxx/` placeholders in the script with actual paths
4. **Multi-GPU Training**: The script defaults to using 4 GPUs (`--nproc_per_node=4`), adjust according to your actual situation

## Project Structure

```
sft_training/
├── readme.md              # This file
├── sft_train.sh           # Training script
├── ds_z3_config.json      # DeepSpeed configuration
├── merge_model.yaml       # Model merging configuration
├── requirements.txt       # Python dependencies
└──  environment.yml        # Conda environment configuration
```

