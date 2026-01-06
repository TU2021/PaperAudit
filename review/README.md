# Review System

Batch paper review system using multi-agent architecture for comprehensive paper evaluation.

## Overview

This system provides two main review agents:
1. **AuditAgent**: Multi-stage review with baseline review, cheating detection, motivation evaluation, and final assessment
2. **DeepReviewerAgent**: Deep review with multiple reviewer perspectives

The system also includes alignment evaluation tools to compare AI-generated reviews with human reviews.

## System Architecture

### Core Components

1. **Review Agents** (`agents/`):
   - `AuditAgent`: Comprehensive multi-stage review agent
   - `DeepReviewerAgent`: Deep review with multiple perspectives

2. **Batch Runners**:
   - `run_audit_agent.py`: Batch runner for AuditAgent
   - `run_deepreview_agent.py`: Batch runner for DeepReviewerAgent

3. **Evaluation Tools** (`alignment/`):
   - `eval_alignment.py`: Compare AI reviews with human reviews
   - `calculate_alignment_score.py`: Aggregate alignment metrics
   - `merge_deepreview_output.py`: Merge multiple review outputs

4. **Automation Scripts**:
   - `run_all_review.sh`: Batch run AuditAgent for multiple models
   - `run_all_deepreview.sh`: Batch run DeepReviewerAgent for multiple models

## File Structure

```
review/
├── agents/              # Review agent implementations
├── alignment/           # Alignment evaluation tools
├── config.yml          # Global configuration (LLM, retrieval, concurrency)
├── run_audit_agent.py     # AuditAgent batch runner
├── run_deepreview_agent.py  # DeepReviewerAgent batch runner
├── run_all_review.sh   # Batch automation for AuditAgent
├── run_all_deepreview.sh   # Batch automation for DeepReviewerAgent
└── requirements.txt    # Python dependencies
```

## Configuration

The system is configured via `config.yml` in the root directory. This file allows you to adjust various parameters such as retry logic, concurrency limits, and model temperatures for different agents.

### Example `config.yml`

```yaml
# Global configuration for paper audit review
llm:
  max_retries: 10
  initial_retry_delay: 1.0
  max_retry_delay: 10.0
  backoff_multiplier: 2.0
  default_temperature: 0.2
  default_max_tokens: 65536

retrieval:
  retry_delay_base: 2
  semantic_scholar_max_retries: 5
  arxiv_max_retries: 5
  max_results: 15

concurrency:
  cheating_detector: 3
  motivation_evaluator: 5

agents:
  motivation_evaluator:
    temperature: 0.2
  paper_memory_summarizer:
    temperature: 0.2
  cheating_detector:
    temperature: 0.2
  summarizer:
    temperature: 0.2
```

## Usage

### 1. AuditAgent Review

AuditAgent performs comprehensive multi-stage review with multiple stages including baseline review, cheating detection, motivation evaluation, and final assessment.

The runner uses async scheduler (asyncio + semaphore) with threadpool executor for parallel processing. It supports resume (skip if output exists), overwrite, and .inprogress file tracking.

**Pipeline Stages:**
1. Baseline Review: Initial review of the paper
2. Paper Memory: Create natural-language memory of the paper
3. Cheating Detection: Detect potential data leakage or test set contamination
4. Motivation Evaluation: Assess the motivation and contribution clarity
5. Final Assessment: Refined review incorporating all previous stages

**Basic Usage:**
```bash
python run_audit_agent.py \
    --input_dir /path/to/papers \
    --model gpt-5-2025-08-07 \
    --review_agent paper_audit \
    --model_tag all \
    --jobs 10
```

**Review Synthetic Papers:**
```bash
python run_audit_agent.py \
    --input_dir /path/to/papers \
    --model gpt-5-2025-08-07 \
    --synth_model gpt-5-2025-08-07 \
    --review_agent paper_audit \
    --model_tag all \
    --jobs 10
```

**Disable Specific Stages:**
```bash
python run_audit_agent.py \
    --input_dir /path/to/papers \
    --model gpt-5-2025-08-07 \
    --no_cheating_detection \
    --no_motivation \
    --jobs 5
```

### 2. DeepReviewerAgent Review

DeepReviewerAgent performs deep, comprehensive paper review with multiple reviewer perspectives. The agent generates detailed reviews with structured scores across multiple dimensions.

The runner uses async scheduler (asyncio + semaphore) with threadpool executor for parallel processing. It supports resume (skip if output exists), overwrite, and .inprogress file tracking.

**Basic Usage:**
```bash
python run_deepreview_agent.py \
    --input_dir /path/to/papers \
    --model gpt-5-2025-08-07 \
    --review_agent deepreviewer \
    --model_tag all \
    --jobs 10
```

**Review Synthetic Papers:**
```bash
python run_deepreview_agent.py \
    --input_dir /path/to/papers \
    --model gpt-5-2025-08-07 \
    --synth_model gpt-5-2025-08-07 \
    --review_agent deepreviewer \
    --model_tag all \
    --jobs 10
```

### 3. Batch Automation

**Run AuditAgent for Multiple Models:**
```bash
# Edit run_all_review.sh to configure models and parameters
bash run_all_review.sh
```

**Run DeepReviewerAgent for Multiple Models:**
```bash
# Edit run_all_deepreview.sh to configure models and parameters
bash run_all_deepreview.sh
```

### 4. Alignment Evaluation

**Merge DeepReview Outputs:**

Batch merge/consolidate DeepReviewer outputs into a single merged review per paper. The merger reads multiple review files and uses LLM to consolidate them into one comprehensive review with only Summary, Strengths, and Weaknesses sections.

```bash
python alignment/merge_deepreview_output.py \
    --input_dir /path/to/papers \
    --review_agent deepreviewer \
    --ai_model gpt-5-2025-08-07 \
    --ai_review_file deep_review_all.txt \
    --out_file deep_review_merge.txt \
    --model gpt-5-2025-08-07 \
    --jobs 10
```

**Evaluate Review Alignment:**

Batch review alignment evaluator that compares AI-generated reviews with human reviews based on COVERAGE alignment (not score alignment). The evaluator compares human review summary with AI review text, handles multiple AI reviews by consolidating them, extracts key points, and computes coverage metrics.

```bash
python alignment/eval_alignment.py \
    --input_dir /path/to/papers \
    --judge_model gemini-2.5-pro \
    --review_agent paper_audit \
    --ai_model gpt-5-2025-08-07 \
    --ai_review_file review_output_all.json \
    --model_tag v1 \
    --jobs 10
```

**Aggregate Alignment Scores:**

Aggregates COVERAGE alignment metrics over a folder of paper subfolders. Reads alignment evaluation results from multiple papers and computes aggregated statistics (mean, std, min, max, count) for each alignment metric.

```bash
# Aggregate all alignment results
python alignment/calculate_alignment_score.py \
    --input_dir /path/to/papers \
    --judge_model gemini-2.5-pro \
    --model_tag v1

# Aggregate with filters
python alignment/calculate_alignment_score.py \
    --input_dir /path/to/papers \
    --judge_model gemini-2.5-pro \
    --model_tag v1 \
    --review_agent paper_audit \
    --ai_model gpt-5-2025-08-07 \
    --ai_review_file review_output_all.json \
    --write_rows
```

## Key Parameters

### Input/Output
- `--input_dir`: Root folder containing paper subfolders
- `--synth_model`: Synthetic data model suffix (empty => origin papers)
- `--model_tag`: User-specified tag for output filenames

### Model Configuration
- `--model`: LLM model name for review
- `--review_agent`: Reviewer agent name (used as output subdir)

### Feature Flags
- `--enable_mm` / `--disable_mm`: Enable/disable multimodal text markers
- `--reuse_cache`: Reuse cached artifacts if present
- `--no_cheating_detection`: Disable cheating detection stage (AuditAgent only)
- `--no_motivation`: Disable motivation evaluation stage (AuditAgent only)

### Batch Processing
- `--jobs`: Number of in-flight papers (parallel processing)
- `--overwrite`: Force re-run and overwrite existing outputs

## Output Structure

### AuditAgent Output
- **Location**: `<paper_dir>/reviews/{review_agent}/{model}/{paper_origin|paper_synth_{synth_model}}/review_output_{model_tag}.json`
- **Content**:
  - `baseline_review`: Initial review text
  - `final_review`: Refined review text
  - `scores`: Parsed scores for baseline and refined reviews
    - `overall`, `novelty`, `technical_quality`, `clarity`, `confidence`
- **Cache Artifacts** (if `--reuse_cache` enabled):
  - `paper_memory.txt`
  - `cheat_report.txt`
  - `motivation_report.txt`
  - `baseline_review.txt`
  - `final_review_*.txt`

### DeepReviewerAgent Output
- **Review Text**: `<paper_dir>/reviews/{review_agent}/{model}/{paper_origin|paper_synth_{synth_model}}/deep_review_{model_tag}.txt`
- **Review Scores**: `<paper_dir>/reviews/{review_agent}/{model}/{paper_origin|paper_synth_{synth_model}}/review_output_{model_tag}.json`
- **Content**:
  - `scores`: Parsed scores with per-reviewer and averaged scores
    - `per_reviewer`: List of individual reviewer scores
    - `avg`: Averaged scores across reviewers
    - `max`: Maximum possible scores
    - `counts`: Number of reviewers for each field
    - `num_reviewers`: Total number of reviewers

### Alignment Evaluation Output
- **Location**: `<paper_dir>/reviews/alignment_judge/{judge_model}/{paper_origin}/alignment_{model_tag}.json`
- **Metrics**:
  - `strength_coverage_recall`: How well AI covers human strengths (0-1)
  - `weakness_coverage_recall`: How well AI covers human weaknesses (0-1)
  - `ai_extra_major_points_rate`: Rate of AI points not in human review (0-1)
  - `symmetric_coverage_similarity`: Jaccard similarity of all points (0-1)

## Workflow

1. **Run Reviews**: Use `run_audit_agent.py` or `run_deepreview_agent.py` to generate reviews
2. **Evaluate Alignment**: Use `alignment/eval_alignment.py` to compare AI reviews with human reviews
3. **Aggregate Metrics**: Use `alignment/calculate_alignment_score.py` to compute aggregate statistics
4. **Merge Outputs** (optional): Use `alignment/merge_deepreview_output.py` to consolidate multiple reviews

## Environment Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables (`.env` file):
   ```
   OPENAI_API_KEY=your-api-key
   # Add other API keys as needed
   ```

3. Configure `config.yml` for agent parameters

## Supported Models

The system supports various LLM models including:
- `o4-mini`
- `gpt-5-2025-08-07`
- `gemini-2.5-pro`
- `claude-sonnet-4-5-20250929`
- `grok-4`
- `deepseek-v3.1`
- And other OpenAI-compatible models

## Notes

- All runners use async scheduler (asyncio + semaphore) with threadpool executor for parallel processing
- Resume support: automatically skip papers that already have output files
- Progress tracking: `.inprogress` files are created during processing
- Error handling: failed papers are logged and reported at the end

