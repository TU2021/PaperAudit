# PaperAudit: Academic Paper Error Detection and Review System

A comprehensive Large Language Model (LLM)-based system for academic paper error detection and review, providing end-to-end solutions from data preprocessing, error detection, paper review to model training.

**[中文](README_cn.md)**

## Project Overview

PaperAudit is a comprehensive academic paper quality assessment system with the following main features:

- **Paper Preprocessing**: Download papers from OpenReview, parse PDFs, and generate synthetic error data
- **Error Detection**: Use Multi-Agent System (MAS) to detect various types of errors in papers
- **Paper Review**: Provide multi-stage, multi-perspective automated paper review
- **Model Training**: Support Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) training

## System Architecture

```
PaperAudit/
├── preprocess_data/    # Data preprocessing pipeline
├── detect/            # Multi-agent error detection system
├── review/            # Multi-agent paper review system
└── train/             # Model training (SFT & RL)
    ├── train_data_process/  # Training data processing
    ├── sft_training/        # Supervised fine-tuning
    ├── rl_training/         # Reinforcement learning training
    └── eval/                # Model evaluation
```

## Core Modules

### 1. preprocess_data - Data Preprocessing

Download papers from OpenReview and generate error detection data for training.

**Main Features:**
- Download paper PDFs, reviews, and metadata from OpenReview API
- Parse PDFs into structured JSON using LlamaParse and LLM
- Add section labels (Abstract, Introduction, Method, etc.)
- Generate 8 categories of synthetic errors (evidence/data manipulation, method logic flaws, experimental design issues, etc.)

**Key Scripts:**
- `download_openreview.py` - Download papers
- `parse_paper.py` - Parse PDFs
- `add_section.py` - Add section labels
- `synth_corruptions_for_detector.py` - Generate synthetic errors

### 2. detect - Error Detection System

Multi-Agent System (MAS) based paper error detection that can identify factual errors, logical inconsistencies, citation errors, and more.

**Main Features:**
- Multi-agent collaborative detection (Planner, Retriever, Specialist)
- Support for multiple detection modes (basic, enhanced, full)
- Detection result evaluation and statistical analysis

**Key Scripts:**
- `mas_error_detection.py` - Main detection pipeline
- `eval_detection.py` - Detection result evaluation
- `eval_log_detail.py` - Detailed statistical analysis

### 3. review - Paper Review System

Provides multi-stage, multi-perspective automated paper review, including baseline review, cheating detection, motivation evaluation, etc.

**Main Features:**
- **AuditAgent**: Multi-stage review (baseline review → cheating detection → motivation evaluation → final assessment)
- **DeepReviewerAgent**: Multi-perspective deep review
- Alignment evaluation between review results and human reviews

**Key Scripts:**
- `run_audit_agent.py` - Batch runner for AuditAgent
- `run_deepreview_agent.py` - Batch runner for DeepReviewerAgent
- `alignment/eval_alignment.py` - Alignment evaluation

### 4. train - Model Training

Supports Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) training for error detection and review models.

**Main Features:**
- **SFT Training**: Parameter-efficient fine-tuning using LLaMA-Factory and LoRA
- **RL Training**: Reinforcement learning training using VERL framework and GRPO algorithm
- **Data Processing**: Training data format conversion and processing
- **Model Evaluation**: Evaluation tools for API models and base models

**Supported Models:**
- Qwen3-8B / Qwen3-14B
- Llama-3.2-3B-Instruct

## Quick Start

### Environment Setup

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure Environment Variables:**
```bash
cp env.example env.sh
# Edit env.sh to set necessary API keys
source env.sh
```

### Usage Examples

#### 1. Data Preprocessing
```bash
cd preprocess_data
# Download papers
python download_openreview.py --conference ICLR.cc --year 2025 --type oral

# Parse PDFs
python parse_paper.py --root-dir ./data/ICLR_2025_oral --model gpt-5-2025-08-07

# Generate synthetic errors
python synth_corruptions_for_detector.py --input-dir ./data --output-dir ./corrupted_data
```

#### 2. Error Detection
```bash
cd detect
# Run detection
python mas_error_detection.py --input-dir ../preprocess_data/output --model gpt-5-2025-08-07

# Evaluate results
python eval_detection.py --detection-dir ./results --ground-truth-dir ../preprocess_data/corrupted_data
```

#### 3. Paper Review
```bash
cd review
# Run AuditAgent
python run_audit_agent.py --input-dir ../data/ICLR_26 --model gpt-5-2025-08-07

# Run DeepReviewerAgent
python run_deepreview_agent.py --input-dir ../data/ICLR_26 --model gpt-5-2025-08-07
```

#### 4. Model Training
```bash
cd train
# SFT training
cd sft_training
bash sft_train.sh

# RL training
cd ../rl_training
bash run_grpo_train.sh
```

## Configuration

### Environment Variables

Configure the following environment variables in `env.sh`:

- `OPENAI_API_KEY` - OpenAI API key
- `LLAMA_API_KEY` - LlamaParse API key
- `OPENREVIEW_USERNAME` - OpenReview username
- `OPENREVIEW_PASSWORD` - OpenReview password

### Configuration Files

- `review/config.yml` - Review system configuration (LLM parameters, concurrency settings, etc.)
- `train/sft_training/ds_z3_config.json` - DeepSpeed configuration
- `train/rl_training/config/grpo_config.yaml` - GRPO training configuration

## Dependencies

Main dependencies include:
- **Web Framework**: FastAPI, Uvicorn
- **LLM API**: OpenAI, LiteLLM
- **Deep Learning**: PyTorch, Transformers, Accelerate
- **Data Processing**: Pandas, NumPy, PyArrow
- **PDF Processing**: PyPDF2, Pillow

See `requirements.txt` for the complete dependency list.

## Project Structure

```
ACL/
├── preprocess_data/          # Data preprocessing
│   ├── download_openreview.py
│   ├── parse_paper.py
│   ├── add_section.py
│   └── synth_corruptions_for_detector.py
├── detect/                   # Error detection
│   ├── mas_error_detection.py
│   ├── agents.py
│   ├── eval_detection.py
│   └── eval_log_detail.py
├── review/                   # Paper review
│   ├── agents/
│   │   ├── PaperAudit/      # AuditAgent
│   │   └── deepreviewer.py  # DeepReviewerAgent
│   ├── alignment/           # Alignment evaluation
│   └── run_audit_agent.py
├── train/                    # Model training
│   ├── train_data_process/   # Data processing
│   ├── sft_training/         # SFT training
│   ├── rl_training/          # RL training
│   └── eval/                 # Model evaluation
├── requirements.txt          # Project dependencies
├── env.example               # Environment variables example
└── README.md                 # This document
```

## Workflow

Typical complete workflow:

1. **Data Preparation**: Use `preprocess_data` to download and preprocess papers, generating synthetic error data
2. **Error Detection**: Use `detect` module to detect errors in papers
3. **Paper Review**: Use `review` module for multi-stage review
4. **Model Training**: Use `train` module to train and improve detection/review models
5. **Evaluation & Optimization**: Evaluate model performance and iterate improvements
