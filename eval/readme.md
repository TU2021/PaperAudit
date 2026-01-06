# Model Evaluation Project

This project provides evaluation scripts for assessing large language models on academic paper error detection tasks. It supports both single model evaluation and batch evaluation of multiple models through external APIs, with automated scoring using a judge model.

## Project Overview

The evaluation system is designed to test models on scientific error detection tasks, where models must identify errors, methodological flaws, or academic integrity issues in paper text. The evaluation uses a judge model (typically Qwen3-80B) to automatically score model predictions against ground truth annotations.

## Features

- **Single Model Evaluation**: Evaluate one model at a time with `eval_base_model.py`
- **Multi-Model Batch Evaluation**: Evaluate multiple models concurrently with `eval_api_model.py`
- **Automatic Scoring**: Uses a judge model to evaluate predictions against ground truth
- **Resume Capability**: JSONL-based incremental saving allows resuming interrupted evaluations
- **Concurrent Processing**: Asynchronous API calls with configurable concurrency limits
- **Error Handling**: Robust retry mechanisms and graceful error handling
- **Token Management**: Automatic token counting and input truncation to fit context limits

## Files

- `eval_base_model.py`: Single model evaluation script
- `eval_api_model.py`: Multi-model batch evaluation script with resume support

## Environment Setup

### Dependencies

Install required Python packages:

```bash
pip install aiohttp tiktoken
```

### Configuration

Both scripts require configuration at the top of the file. Key parameters include:

#### Common Configuration

- `TEST_FILE_PATH`: Path to test dataset (JSON array format)
- `JUDGE_MODEL_NAME`: Name/identifier of the judge model
- `JUDGE_API_KEY`: API key for judge model (set to "EMPTY" if not required)
- `JUDGE_BASE_URL`: Base URL for judge model API server
- `JUDGE_TEMPERATURE`: Temperature for judge model (typically 0.0)
- `MODEL_MAX_CONTEXT_TOKENS`: Maximum context length for evaluated models
- `JUDGE_MODEL_MAX_CONTEXT_TOKENS`: Maximum context length for judge model
- `TOKEN_BUFFER`: Buffer tokens reserved for safety
- `MAX_INPUT_TRUNCATE_LENGTH`: Maximum input length before truncation
- `RETRY_TIMES`: Number of retry attempts for failed requests
- `REQUEST_TIMEOUT`: Request timeout in seconds

#### eval_base_model.py Specific

- `MODEL_NAME`: Name of the model to evaluate
- `MODEL_API_KEY`: API key for the model
- `MODEL_BASE_URL`: Base URL for model API
- `MODEL_TEMPERATURE`: Temperature for model inference
- `MAX_CONCURRENT_REQUESTS`: Maximum concurrent API requests

#### eval_api_model.py Specific

- `EXTERNAL_BASE_URL`: Base URL for external model APIs
- `EXTERNAL_API_KEY`: API key for external models (can be set via environment variable)
- `MODELS_TO_TEST`: List of model names to evaluate
- `OUTPUT_DIR`: Directory for output files
- `MAX_EXTERNAL_INFLIGHT`: Maximum concurrent requests to external models
- `MAX_JUDGE_INFLIGHT`: Maximum concurrent requests to judge model
- `MODEL_WORKERS_PER_MODEL`: Number of worker threads per model
- `RERUN_FAILED_PRED`: Whether to retry failed predictions

## Usage

### Single Model Evaluation

1. **Configure the script**: Edit `eval_base_model.py` and set all configuration parameters at the top of the file.

2. **Prepare test data**: Ensure your test file is a JSON array with each sample containing:
   ```json
   {
     "instruction": "Task prompt/instruction",
     "output": "[{\"error_location\": \"...\", \"error_type\": \"...\", \"error_explanation\": \"...\"}]"
   }
   ```

3. **Run evaluation**:
   ```bash
   python eval_base_model.py
   ```

4. **Results**: The script outputs a JSON file with evaluation results for each sample.

### Multi-Model Batch Evaluation

1. **Set environment variable** (if using external API):
   ```bash
   export EXTERNAL_API_KEY='your_api_key'
   ```

2. **Configure the script**: Edit `eval_api_model.py` and set all configuration parameters.

3. **Prepare test data**: Same format as single model evaluation.

4. **Run evaluation**:
   ```bash
   python eval_api_model.py
   ```

5. **Results**: 
   - JSONL files: One per model (`{model_name}.jsonl`) with incremental results
   - JSON files: Merged final results (`{model_name}.json`) after completion

### Resuming Interrupted Evaluation

The `eval_api_model.py` script supports resuming interrupted evaluations:

1. **Automatic Resume**: Simply run the script again with the same configuration. It will:
   - Load existing JSONL state files
   - Skip already completed samples
   - Continue from where it left off

2. **Manual Resume**: The script checks for existing JSONL files and automatically resumes based on:
   - `pred_done`: Whether prediction was completed
   - `pred_success`: Whether prediction was successful
   - `judge_done`: Whether judgment was completed

## Data Format

### Input Format

Test files should be JSON arrays with the following structure:

```json
[
  {
    "instruction": "Please identify errors in the following paper text: ...",
    "output": "[{\"error_location\": \"Section 2, paragraph 3\", \"error_type\": \"methodological\", \"error_explanation\": \"...\"}]"
  },
  ...
]
```

- `instruction`: The task prompt/instruction for the model
- `output`: Ground truth as a JSON string (array of error objects)

### Output Format

Each evaluation result contains:

```json
{
  "sample_index": 1,
  "model": "model-name",
  "status": "success",
  "instruction": "...",
  "prediction": "[{\"error_location\": \"...\", ...}]",
  "ground_truth": [...],
  "judge_result": {
    "matches": [...],
    "overall_match_score": 10
  }
}
```

## Judge Model Evaluation

The judge model evaluates predictions based on:

1. **Match Detection**: Compares each ground truth error with model predictions
2. **Semantic Matching**: Checks if predictions match ground truth in terms of:
   - Error type classification
   - Error location (overlap with ground truth)
   - Error explanation (semantic similarity)
3. **Overall Score**: Provides an overall match score (typically 0-10)

The judge prompt is automatically constructed to compare ground truth annotations with model predictions and return structured JSON results.

## Token Management

Both scripts include automatic token management:

- **Token Counting**: Uses `tiktoken` with `cl100k_base` encoding
- **Input Truncation**: Automatically truncates long inputs to fit context limits
- **Completion Budget**: Calculates available tokens for completion based on input length
- **Safety Buffers**: Reserves token buffer to prevent context overflow

## Concurrency Control

### eval_base_model.py

- Uses a semaphore to limit concurrent requests
- Default: `MAX_CONCURRENT_REQUESTS = 32`

### eval_api_model.py

- Separate semaphores for external models and judge model
- Configurable workers per model
- Default: `MAX_EXTERNAL_INFLIGHT = 40`, `MAX_JUDGE_INFLIGHT = 64`

## Error Handling

Both scripts include robust error handling:

- **Retry Logic**: Automatic retries for transient failures (429 rate limits, timeouts)
- **Rate Limiting**: Exponential backoff for rate limit errors
- **Timeout Handling**: Configurable request timeouts with retries
- **Invalid Samples**: Graceful handling of malformed or invalid test samples
- **API Failures**: Detailed error messages and fallback handling

## Performance Optimization

- **Asynchronous I/O**: Uses `aiohttp` for concurrent API calls
- **Connection Pooling**: Reuses HTTP connections for efficiency
- **Incremental Saving**: JSONL format allows real-time progress tracking
- **State Management**: Efficient state tracking for resume functionality

## Output Files

### eval_base_model.py

- Single JSON file with all evaluation results

### eval_api_model.py

- **JSONL files** (`{model_name}.jsonl`): Incremental logs with:
  - `record_type`: "pred" or "judge"
  - `sample_index`: Sample identifier
  - `status`: "success" or "failed"
  - `prediction`: Model output
  - `judge_result`: Judge model evaluation (for judge records)

- **JSON files** (`{model_name}.json`): Final merged results (one record per sample)

## Notes

1. **Judge Model Server**: Ensure the judge model API server is running before evaluation
2. **API Compatibility**: Both scripts use OpenAI-compatible API endpoints (`/chat/completions`)
3. **Data Validation**: Scripts automatically filter invalid samples (missing instruction/output)
4. **Ground Truth Format**: Ground truth must be a valid JSON array string in the `output` field
5. **Network Requirements**: Ensure stable network connectivity for API calls
6. **Resource Usage**: Adjust concurrency limits based on API rate limits and available resources

## Troubleshooting

### Common Issues

1. **API Connection Errors**:
   - Verify API endpoints are accessible
   - Check API keys and authentication
   - Ensure judge model server is running

2. **Rate Limiting**:
   - Reduce `MAX_EXTERNAL_INFLIGHT` or `MAX_JUDGE_INFLIGHT`
   - Increase retry delays
   - Check API rate limits

3. **Context Length Errors**:
   - Reduce `MAX_INPUT_TRUNCATE_LENGTH`
   - Increase `MODEL_MAX_CONTEXT_TOKENS` if possible
   - Check token counting accuracy

4. **Resume Not Working**:
   - Verify JSONL files exist in output directory
   - Check file permissions
   - Ensure same configuration is used for resume

5. **Invalid JSON from Judge**:
   - Check judge model prompt format
   - Verify judge model is functioning correctly
   - Review judge model output for parsing errors

## Example Workflow

1. **Start judge model server** (using vLLM or similar):
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model /path/to/judge/model \
     --port 8001
   ```

2. **Configure evaluation script**:
   - Set `JUDGE_BASE_URL = "http://localhost:8001/v1"`
   - Set `TEST_FILE_PATH` to your test data
   - Configure model API endpoints

3. **Run evaluation**:
   ```bash
   python eval_api_model.py
   ```

4. **Monitor progress**: Check JSONL files for real-time results

5. **Analyze results**: Use merged JSON files for final analysis

