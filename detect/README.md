# MAS Error Detection System

A comprehensive multi-agent system (MAS) for detecting errors in academic papers using Large Language Models (LLMs). The system employs multiple specialized agents working together to identify various types of errors, including factual inaccuracies, logical inconsistencies, citation errors, and cross-section issues.

## Overview

This detection system processes academic papers (in JSON format from the preprocessing pipeline) and uses a multi-stage approach to identify errors.

**Execution Order**: The pipeline follows a three-step workflow: (1) Run `mas_error_detection.py` to generate detection results, (2) Run `eval_detection.py` to match findings against ground truth and evaluate detection performance, (3) Run `eval_log_detail.py` to compute quantitative statistics and breakdowns.

## System Architecture

### Core Components

- **`mas_error_detection.py`**: Main detection pipeline script
- **`agents.py`**: All agent-related functions (Planner, Retriever, Specialist, etc.)
- **`prompts.py`**: Centralized prompt templates for all LLM agents
- **`mas_reference_helper.py`**: Reference extraction and enrichment utilities
- **`utils.py`**: Common utility functions

### Evaluation Components

- **`eval_detection.py`**: Evaluates detection results by matching findings against ground truth
- **`eval_log_detail.py`**: Computes detailed breakdown statistics (macro/micro rates by corruption type, difficulty, location, etc.)

### Batch Processing Scripts

- **`run_all_detect.sh`**: Batch script for running detection across multiple models
- **`run_all_eval.sh`**: Batch script for running evaluation across multiple configurations

## File Structure

```
detect/
├── mas_error_detection.py      # Main detection pipeline
├── agents.py                   # Agent functions (Planner, Retriever, Specialist, etc.)
├── prompts.py                  # LLM prompt templates
├── mas_reference_helper.py     # Reference extraction and enrichment
├── eval_detection.py           # Evaluation script (GT vs findings matching)
├── eval_log_detail.py          # Detailed statistics computation
├── utils.py                    # Utility functions
├── run_all_detect.sh           # Batch detection script
├── run_all_eval.sh             # Batch evaluation script
└── README.md                   # This file
```

## Detection Modes

The system supports three main detection modes with different feature combinations:

### Fast Mode
Minimal features for quick evaluation:
- Global review (optional)
- Basic per-task pipeline
- No memory, no section review, no retriever, no merge

### Standard Mode
Balanced features:
- Section review
- Memory building
- Per-task pipeline
- No global review, no retriever, no merge

### Deep Mode
Comprehensive features for thorough evaluation:
- All features enabled: global review, section review, per-task, memory, retriever, web search, merge
- Maximum context and evidence gathering

## Usage

### Basic Detection

```bash
# Fast detection mode
python mas_error_detection.py \
    --root_dir /path/to/papers \
    --detect_mode fast \
    --detect_model o4-mini \
    --synth_model gpt-5-2025-08-07 \
    --enable-global-review \
    --jobs 10

# Standard detection mode
python mas_error_detection.py \
    --root_dir /path/to/papers \
    --detect_mode standard \
    --detect_model o4-mini \
    --synth_model gpt-5-2025-08-07 \
    --enable-memory-build \
    --enable-section-review \
    --jobs 10

# Deep detection mode (all features)
python mas_error_detection.py \
    --root_dir /path/to/papers \
    --detect_mode deep \
    --detect_model o4-mini \
    --synth_model gpt-5-2025-08-07 \
    --enable-memory-build \
    --enable-section-review \
    --enable-per-task \
    --enable-retriever \
    --enable-memory-injection \
    --enable-merge \
    --jobs 10
```

### Single File Detection

```bash
python mas_error_detection.py \
    --synth_json /path/to/paper_synth_xxx.json \
    --detect_model o4-mini \
    --detect_mode standard \
    --enable-per-task \
    --enable-global-review
```

### Evaluation

```bash
# Evaluate detection results
python eval_detection.py \
    --root_dir /path/to/papers \
    --detect_model o4-mini \
    --synth_model gpt-5-2025-08-07 \
    --eval_mode fast \
    --eval_model gpt-5.1 \
    --concurrency 10 \
    --overwrite

# Compute detailed statistics
python eval_log_detail.py \
    --root_dir /path/to/papers \
    --synth_model gpt-5-2025-08-07 \
    --detect_model o4-mini \
    --eval_mode standard \
    --eval_model gpt-5.1
```

### Batch Processing

```bash
# Run detection for multiple model combinations
bash run_all_detect.sh

# Run evaluation for multiple configurations
bash run_all_eval.sh
```

## Key Parameters

### Input/Output
- `--root_dir`: Root directory for batch processing (contains `paper_synth_*.json` files)
- `--synth_json`: Single file mode - path to a single synth JSON
- `--synth_glob`: Glob pattern to find synth JSONs (default: `paper_synth_*.json`)
- `--overwrite`: Force overwrite existing outputs
- `--overwrite_zero`: Rerun if output exists but has empty findings

### Model Configuration
- `--detect_model`: Model tag for detection (e.g., `o4-mini`, `gemini-2.5-pro`)
- `--eval_model`: Model tag for evaluation/adjudication (e.g., `gpt-5.1`)
- `--synth_model`: Filter papers by synthesis model name
- `--detect_mode`: Mode name used in output paths (e.g., `fast`, `standard`, `deep`)

### Feature Flags
All feature flags default to `False` and must be explicitly enabled:

- `--enable-global-review`: Enable global cross-section review
- `--enable-section-review`: Enable section-level review
- `--enable-per-task`: Enable per-task pipeline (Planner → Retriever → Specialist)
- `--enable-retriever`: Enable Retriever for evidence extraction
- `--enable-web-search`: Enable web search after Retriever
- `--enable-memory-build`: Enable full-paper memory construction
- `--enable-memory-injection`: Inject memory into Specialist
- `--enable-section-findings-as-prior`: Pass section findings to Specialist as prior
- `--enable-merge`: Enable merging/adjudication of findings
- `--enable-mm`: Enable multimodal input (images) - Default: `True`

### Performance Settings
- `--jobs`: Thread pool size for parallel processing (default: 10)
- `--concurrency`: Number of concurrent evaluation tasks (default: 8)
- `--max_ctx_chars`: Maximum context characters for LLM input (default: 120000)
- `--memory_max_chars`: Max characters of memory injected per task (default: 20000)
- `--empty_retry_times`: Number of retries for empty output (default: 2)

## Output Structure

All detection and evaluation results are organized under a `detect/` subdirectory:

```
<paper_dir>/
└── detect/
    ├── <detect_mode>_detect/
    │   └── <detect_model>/
    │       └── <synth_stem>/
    │           └── <detect_mode>_detect.json          # Detection results
    ├── <detect_mode>_detect_log/
    │   └── <detect_model>/
    │       └── <synth_stem>/
    │           ├── ground_truth.from_synth.json      # Extracted GT
    │           ├── paper.blocks.head.json            # Paper blocks sample
    │           ├── outline.json                      # Paper outline
    │           ├── memory.meta.json                  # Memory metadata
    │           ├── global_review.meta.json           # Global review metadata
    │           ├── section_review.meta.json          # Section review metadata
    │           ├── per_task.meta.json               # Per-task metadata
    │           └── task_*/                           # Individual task directories
    └── <detect_mode>_eval/
        └── <detect_model>/
            └── <synth_stem>/
                └── eval_<eval_model>.json            # Evaluation results
```

### Detection Output Format

The detection JSON (`<detect_mode>_detect.json`) contains:
- `findings`: List of detected errors with:
  - `id`: Unique identifier
  - `corruption_type`: Type of error
  - `difficulty`: Difficulty level
  - `location`: Section/location of error
  - `description`: Detailed description
  - `evidence`: Supporting evidence
  - `needs_cross_section`: Whether cross-section analysis is needed

### Evaluation Output Format

The evaluation JSON (`eval_<eval_model>.json`) contains:
- `matches`: List of matches between GT and findings
- `gt_count`: Total ground truth errors
- `matched_count`: Number of matched findings
- `detection_rate`: Recall (matched / gt_count)
- `precision`: Precision metric
- `f1_score`: F1 score

### Statistics Output

The `eval_log_detail.json` contains aggregated statistics:
- Overall detection rates (macro and micro)
- Breakdown by:
  - `corruption_type`: Error type distribution
  - `difficulty`: Difficulty level distribution
  - `location`: Location/section distribution
  - `needs_cross_section`: Cross-section requirement distribution

## Workflow

### Detection Pipeline

1. **Input**: Paper JSON files from preprocessing pipeline
2. **Memory Building** (if enabled): Create natural-language memory of full paper
3. **Global Review** (if enabled): Cross-section error detection
4. **Section Review** (if enabled): Section-level error detection
5. **Task Planning** (if enabled): Generate focused tasks
6. **Evidence Retrieval** (if enabled): Extract evidence and generate queries
7. **Web Search** (if enabled): Search for additional evidence
8. **Specialist Review**: Review each task with context and evidence
9. **Merging** (if enabled): Merge and adjudicate findings
10. **Output**: Detection results JSON

### Evaluation Pipeline

1. **Input**: Detection results and ground truth from synth JSON
2. **Matching**: Use LLM to match findings against GT
3. **Statistics**: Compute detection rates and breakdowns
4. **Output**: Evaluation JSON and aggregated statistics

## Supported Models

### Detection Models
- `gpt-5-2025-08-07`
- `o4-mini`
- `gemini-2.5-pro`
- `claude-sonnet-4-5-20250929`
- `grok-4`
- `doubao-seed-1-6-251015`
- `glm-4.6`
- `deepseek-v3.1`
- `kimi-k2-250905`
- `qwen3-235b-a22b-instruct-2507`
- `qwen3-vl-235b-a22b-instruct`

### Evaluation Models
Same as detection models, plus:
- `gpt-5.1` (commonly used for evaluation)

## Environment Requirements

- Python 3.x
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Required Python packages (see project dependencies)

## Notes

- All detection and evaluation results are stored under `<paper_dir>/detect/` directory
- The system supports multimodal input (images) by default
- Memory injection requires memory building to be enabled
- Web search requires retriever to be enabled
- Section findings as prior requires section review to be enabled
- The system uses retry mechanisms for empty LLM outputs
- Batch processing scripts support multiple model combinations

## Related Documentation

- See `../preprocess_data/README.md` for preprocessing pipeline documentation
- Individual script docstrings contain detailed parameter descriptions
