# PaperAudit: Academic Paper Error Detection and Review System

一个基于大语言模型（LLM）的学术论文错误检测与审查系统，提供从数据预处理、错误检测、论文审查到模型训练的全流程解决方案。

<p align="center">
  <a href="https://github.com/TU2021/PaperAudit"><strong>💻 代码</strong></a> •
  <a href="https://huggingface.co/datasets/mayiwen/PaperAudit_Dataset"><strong>📄 论文</strong></a> •
  <a href="https://huggingface.co/datasets/mayiwen/PaperAudit_Dataset"><strong>📊 数据集</strong></a> •
  <a href="https://huggingface.co/mayiwen/PaperAudit_Models"><strong>🤖 模型</strong></a> •
  <a href="README.md"><strong>🇬🇧 English</strong></a>
</p>

## 项目概述

PaperAudit 是一个综合性的学术论文质量评估系统，主要功能包括：

- **论文预处理**：从 OpenReview 下载论文，解析 PDF，生成合成错误数据
- **错误检测**：使用多智能体系统（MAS）检测论文中的各类错误
- **论文审查**：提供多阶段、多视角的自动化论文审查
- **模型训练**：支持监督微调（SFT）和强化学习（RL）训练

## 系统架构

```
PaperAudit/
├── preprocess_data/    # 数据预处理管道
├── detect/            # 多智能体错误检测系统
├── review/            # 多智能体论文审查系统
└── train/             # 模型训练（SFT & RL）
    ├── train_data_process/  # 训练数据处理
    ├── sft_training/        # 监督微调训练
    ├── rl_training/         # 强化学习训练
    └── eval/                # 模型评估
```

## 核心模块

### 1. preprocess_data - 数据预处理

从 OpenReview 下载论文并生成用于训练的错误检测数据。

<p align="center">
  <img src="figs/1_benchmark.pdf" alt="Benchmark 构造" width="800"/>
</p>

**主要功能：**
- 从 OpenReview API 下载论文 PDF、评审和元数据
- 使用 LlamaParse 和 LLM 解析 PDF 为结构化 JSON
- 添加章节标签（Abstract, Introduction, Method 等）
- 生成 8 类合成错误（证据/数据操纵、方法逻辑缺陷、实验设计问题等）

**关键脚本：**
- `download_openreview.py` - 下载论文
- `parse_paper.py` - 解析 PDF
- `add_section.py` - 添加章节标签
- `synth_corruptions_for_detector.py` - 生成合成错误

### 2. detect - 错误检测系统

基于多智能体系统（MAS）的论文错误检测，能够识别事实错误、逻辑不一致、引用错误等问题。

<p align="center">
  <img src="figs/2a_detect_workflow.pdf" alt="错误检测工作流程" width="800"/>
</p>

**主要功能：**
- 多智能体协作检测（Planner, Retriever, Specialist）
- 支持多种检测模式（基础、增强、完整）
- 检测结果评估与统计分析

**关键脚本：**
- `mas_error_detection.py` - 主检测流程
- `eval_detection.py` - 检测结果评估
- `eval_log_detail.py` - 详细统计分析

### 3. review - 论文审查系统

提供多阶段、多视角的自动化论文审查，包括基线审查、作弊检测、动机评估等。

<p align="center">
  <img src="figs/2b_review_workflow.pdf" alt="论文审查工作流程" width="800"/>
</p>

**主要功能：**
- **AuditAgent**：多阶段审查（基线审查 → 作弊检测 → 动机评估 → 最终评估）
- **DeepReviewerAgent**：多视角深度审查
- 审查结果与人工评审的对齐评估

**关键脚本：**
- `run_audit_agent.py` - AuditAgent 批量运行
- `run_deepreview_agent.py` - DeepReviewerAgent 批量运行
- `alignment/eval_alignment.py` - 对齐评估

### 4. train - 模型训练

支持监督微调（SFT）和强化学习（RL）训练，用于训练错误检测和审查模型。

**主要功能：**
- **SFT 训练**：使用 LLaMA-Factory 和 LoRA 进行参数高效微调
- **RL 训练**：使用 VERL 框架和 GRPO 算法进行强化学习训练
- **数据处理**：训练数据格式转换和处理
- **模型评估**：API 模型和基础模型的评估工具

**支持的模型：**
- Qwen3-8B / Qwen3-14B
- Llama-3.2-3B-Instruct

## 快速开始

### 环境配置

1. **安装依赖：**
```bash
pip install -r requirements.txt
```

2. **配置环境变量：**
```bash
cp env.example env.sh
# 编辑 env.sh 设置必要的 API keys
source env.sh
```

### 使用示例

#### 1. 数据预处理
```bash
cd preprocess_data
# 下载论文
python download_openreview.py --conference ICLR.cc --year 2025 --type oral

# 解析 PDF
python parse_paper.py --root-dir ./data/ICLR_2025_oral --model gpt-5-2025-08-07

# 生成合成错误
python synth_corruptions_for_detector.py --input-dir ./data --output-dir ./corrupted_data
```

#### 2. 错误检测
```bash
cd detect
# 运行检测
python mas_error_detection.py --input-dir ../preprocess_data/output --model gpt-5-2025-08-07

# 评估结果
python eval_detection.py --detection-dir ./results --ground-truth-dir ../preprocess_data/corrupted_data
```

#### 3. 论文审查
```bash
cd review
# 运行 AuditAgent
python run_audit_agent.py --input-dir ../data/ICLR_26 --model gpt-5-2025-08-07

# 运行 DeepReviewerAgent
python run_deepreview_agent.py --input-dir ../data/ICLR_26 --model gpt-5-2025-08-07
```

#### 4. 模型训练
```bash
cd train
# SFT 训练
cd sft_training
bash sft_train.sh

# RL 训练
cd ../rl_training
bash run_grpo_train.sh
```

## 配置说明

### 环境变量

在 `env.sh` 中配置以下环境变量：

- `OPENAI_API_KEY` - OpenAI API 密钥
- `LLAMA_API_KEY` - LlamaParse API 密钥
- `OPENREVIEW_USERNAME` - OpenReview 用户名
- `OPENREVIEW_PASSWORD` - OpenReview 密码

### 配置文件

- `review/config.yml` - 审查系统配置（LLM 参数、并发设置等）
- `train/sft_training/ds_z3_config.json` - DeepSpeed 配置
- `train/rl_training/config/grpo_config.yaml` - GRPO 训练配置

## 依赖说明

主要依赖包括：
- **Web 框架**：FastAPI, Uvicorn
- **LLM API**：OpenAI, LiteLLM
- **深度学习**：PyTorch, Transformers, Accelerate
- **数据处理**：Pandas, NumPy, PyArrow
- **PDF 处理**：PyPDF2, Pillow

完整依赖列表请参见 `requirements.txt`。

## 项目结构

```
ACL/
├── preprocess_data/          # 数据预处理
│   ├── download_openreview.py
│   ├── parse_paper.py
│   ├── add_section.py
│   └── synth_corruptions_for_detector.py
├── detect/                   # 错误检测
│   ├── mas_error_detection.py
│   ├── agents.py
│   ├── eval_detection.py
│   └── eval_log_detail.py
├── review/                   # 论文审查
│   ├── agents/
│   │   ├── PaperAudit/      # AuditAgent
│   │   └── deepreviewer.py  # DeepReviewerAgent
│   ├── alignment/           # 对齐评估
│   └── run_audit_agent.py
├── train/                    # 模型训练
│   ├── train_data_process/   # 数据处理
│   ├── sft_training/         # SFT 训练
│   ├── rl_training/          # RL 训练
│   └── eval/                 # 模型评估
├── requirements.txt          # 项目依赖
├── env.example               # 环境变量示例
└── README.md                 # 本文档
```

## 工作流程

典型的完整工作流程：

1. **数据准备**：使用 `preprocess_data` 下载并预处理论文，生成合成错误数据
2. **错误检测**：使用 `detect` 模块检测论文中的错误
3. **论文审查**：使用 `review` 模块进行多阶段审查
4. **模型训练**：使用 `train` 模块训练和改进检测/审查模型
5. **评估优化**：评估模型性能并迭代改进


