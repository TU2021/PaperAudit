import json
import re
import asyncio
import aiohttp
from typing import Dict, Any, List

import tiktoken

# -------------------------- Configuration --------------------------
TEST_FILE_PATH = ""

MODEL_NAME = ""
MODEL_API_KEY = "EMPTY"
MODEL_BASE_URL = ""
MODEL_TEMPERATURE = 0.0

JUDGE_MODEL_NAME = ""
JUDGE_API_KEY = "EMPTY"
JUDGE_BASE_URL = ""
JUDGE_TEMPERATURE = 0.0

MODEL_MAX_CONTEXT_TOKENS = 8192
JUDGE_MODEL_MAX_CONTEXT_TOKENS = 8192
TOKEN_BUFFER = 200
MAX_INPUT_TRUNCATE_LENGTH = 7000
MAX_CONCURRENT_REQUESTS = 32
RETRY_TIMES = 3
REQUEST_TIMEOUT = 300  # seconds


# -------------------------- Data / Token Utilities --------------------------
def load_test_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSON array and keep only valid samples with 'instruction' and 'output'."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Test file must be a JSON array (list of samples).")

        valid = [s for s in data if isinstance(s, dict) and "instruction" in s and "output" in s]
        if len(valid) < len(data):
            print(f"Warning: Filtered out {len(data) - len(valid)} invalid samples.")
        return valid

    except FileNotFoundError as e:
        raise ValueError(f"Test file not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Test file is not valid JSON: {file_path}") from e
    except Exception as e:
        raise ValueError(f"Failed to load test file: {str(e)}") from e


def estimate_token_count(text: str) -> int:
    """Estimate token count using cl100k_base; fallback to a rough heuristic on failure."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        fallback = int(len(text) // 3.5)
        print(f"Warning: Token estimation fallback used ({e}); estimate={fallback}")
        return fallback


def truncate_long_input(text: str, max_tokens: int) -> str:
    """Truncate input to max_tokens (used only when needed) and try to keep basic JSON closure."""
    token_count = estimate_token_count(text)
    if token_count <= max_tokens:
        return text

    encoding = tiktoken.get_encoding("cl100k_base")
    truncated = encoding.decode(encoding.encode(text)[:max_tokens])

    if truncated.startswith("[") and not truncated.endswith("]"):
        truncated += "]"
    if truncated.startswith("{") and not truncated.endswith("}"):
        truncated += "}"

    print(f"Warning: Truncated input from {token_count} to {max_tokens} tokens.")
    return truncated


# -------------------------- Model API Calls --------------------------
async def async_call_model_api(
    session: aiohttp.ClientSession,
    prompt: str,
    model_name: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.0,
    max_context_tokens: int = 8192,
    retry: int = 0,
) -> str:
    """Call OpenAI-compatible /chat/completions endpoint with retries and safe token budgeting."""
    prompt = truncate_long_input(prompt, MAX_INPUT_TRUNCATE_LENGTH)
    input_tokens = estimate_token_count(prompt)

    max_completion_tokens = max_context_tokens - input_tokens - TOKEN_BUFFER
    max_completion_tokens = max(100, min(max_completion_tokens, 2000))

    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key != "EMPTY" else "",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_completion_tokens,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None,
    }

    try:
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        ) as response:
            if response.status == 429:
                if retry < RETRY_TIMES:
                    sleep_time = 2 ** retry
                    print(f"Warning: Rate limited (retry {retry + 1}/{RETRY_TIMES}), sleeping {sleep_time}s")
                    await asyncio.sleep(sleep_time)
                    return await async_call_model_api(
                        session, prompt, model_name, api_key, base_url, temperature, max_context_tokens, retry + 1
                    )
                raise RuntimeError(f"Rate limit exceeded (used {RETRY_TIMES} retries).")

            if response.status != 200:
                detail = await response.text()
                raise RuntimeError(f"API request failed (status {response.status}): {detail[:500]}")

            result = await response.json()
            if not result.get("choices"):
                raise RuntimeError("Empty response from model API.")

            return result["choices"][0]["message"]["content"].strip()

    except asyncio.TimeoutError:
        if retry < RETRY_TIMES:
            print(f"Warning: Request timeout (retry {retry + 1}/{RETRY_TIMES}).")
            await asyncio.sleep(3)
            return await async_call_model_api(
                session, prompt, model_name, api_key, base_url, temperature, max_context_tokens, retry + 1
            )
        raise RuntimeError(f"Request timed out after {RETRY_TIMES + 1} attempts.")

    except Exception as e:
        if retry < RETRY_TIMES:
            print(f"Warning: API error (retry {retry + 1}/{RETRY_TIMES}): {str(e)[:200]}")
            await asyncio.sleep(1)
            return await async_call_model_api(
                session, prompt, model_name, api_key, base_url, temperature, max_context_tokens, retry + 1
            )
        raise RuntimeError(f"API call failed (used {RETRY_TIMES} retries): {str(e)[:500]}")


# -------------------------- Judge Prompt / Parsing --------------------------
def construct_judge_prompt(ground_truth: List[Dict[str, Any]], prediction: str) -> str:
    """Build a strict JSON-only evaluation prompt for the judge model."""
    return f"""
You are a strict but fair professional evaluator for academic paper error detection tasks.
Your only task is to compare the ground truth error annotations with the model's prediction and output a valid JSON evaluation result.

## Evaluation Criteria:
1. error_location: Semantic exact match (minor formatting differences like line breaks are allowed)
2. error_type: String exact match (case-sensitive, must match exactly)
3. error_explanation: Core semantic match (minor wording adjustments are allowed)

## Output Requirements:
- Output ONLY valid JSON (no extra text, no explanations, no markdown)
- overall_match_score: Integer between 0-10 (0 = no match, 10 = perfect match)
- rationale: 2-4 concise sentences explaining the match/mismatch reason
- gt_index: Index of ground truth item (starting from 0)
- matched: Boolean (True = matched, False = not matched)
- matched_pred_indices: List of indices from prediction that match (empty if no match)

## Ground Truth (error annotations):
{json.dumps(ground_truth, ensure_ascii=False, indent=2)}

## Model Prediction (raw output):
{prediction}

## Example Valid Output:
{{
  "matches": [
    {{
      "gt_index": 0,
      "matched": true,
      "matched_pred_indices": [0],
      "rationale": "The error_location matches semantically, error_type is exactly the same, and error_explanation captures the core semantic meaning."
    }}
  ],
  "overall_match_score": 10
}}
""".strip()


async def async_judge_output(
    session: aiohttp.ClientSession,
    ground_truth: List[Dict[str, Any]],
    prediction: str,
) -> Dict[str, Any]:
    """Run judge model and parse a JSON object containing 'matches' and 'overall_match_score'."""
    judge_prompt = construct_judge_prompt(ground_truth, prediction)
    judge_raw_output = await async_call_model_api(
        session=session,
        prompt=judge_prompt,
        model_name=JUDGE_MODEL_NAME,
        api_key=JUDGE_API_KEY,
        base_url=JUDGE_BASE_URL,
        temperature=JUDGE_TEMPERATURE,
        max_context_tokens=JUDGE_MODEL_MAX_CONTEXT_TOKENS,
    )

    json_match = re.search(r"\{[\s\S]*\}", judge_raw_output)
    if not json_match:
        raise RuntimeError(f"No JSON found in judge output. Preview: {judge_raw_output[:500]}")

    try:
        judge_result = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse judge JSON: {e}. Preview: {judge_raw_output[:500]}") from e

    if not isinstance(judge_result, dict) or "matches" not in judge_result or "overall_match_score" not in judge_result:
        raise RuntimeError(f"Judge output missing required keys. Keys: {list(judge_result.keys())}")

    return judge_result


# -------------------------- Sample Processing --------------------------
async def process_sample(
    session: aiohttp.ClientSession,
    sample_idx: int,
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    """Run prediction + judge evaluation for one sample; return a structured result."""
    try:
        print(f"\n=== Processing Sample {sample_idx} ===")
        instruction = str(sample.get("instruction", "")).strip()
        reference_output = str(sample.get("output", "")).strip()

        if not instruction:
            raise ValueError("Empty instruction field.")
        if not reference_output:
            raise ValueError("Empty output (ground truth) field.")

        try:
            ground_truth = json.loads(reference_output)
            if not isinstance(ground_truth, list):
                raise ValueError("Ground truth must be a JSON array.")
        except json.JSONDecodeError as e:
            raise ValueError("Ground truth is not valid JSON.") from e

        print(f"Sample {sample_idx}: Calling small model...")
        model_prediction_text = await async_call_model_api(
            session=session,
            prompt=instruction,
            model_name=MODEL_NAME,
            api_key=MODEL_API_KEY,
            base_url=MODEL_BASE_URL,
            temperature=MODEL_TEMPERATURE,
            max_context_tokens=MODEL_MAX_CONTEXT_TOKENS,
        )
        if not model_prediction_text:
            raise ValueError("Empty prediction from small model.")

        print(f"Sample {sample_idx}: Prediction preview:\n{model_prediction_text[:200]}...")

        print(f"Sample {sample_idx}: Calling judge model...")
        judge_result = await async_judge_output(session, ground_truth, model_prediction_text)

        return {
            "sample_index": sample_idx,
            "status": "success",
            "instruction": instruction[:300] + "..." if len(instruction) > 300 else instruction,
            "prediction": model_prediction_text,
            "judge_result": judge_result,
            "ground_truth_sample": ground_truth[:1],
        }

    except Exception as e:
        err = str(e)[:500]
        print(f"Sample {sample_idx}: Failed - {err}")
        return {
            "sample_index": sample_idx,
            "status": "failed",
            "error": err,
            "prediction": locals().get("model_prediction_text", "N/A"),
            "instruction": (sample.get("instruction", "")[:300] + "...") if sample.get("instruction") else "N/A",
        }


# -------------------------- Main --------------------------
async def main() -> None:
    print("=" * 50)
    print("Starting Academic Paper Error Detection Evaluation")
    print("=" * 50)

    try:
        test_samples = load_test_file(TEST_FILE_PATH)
    except Exception as e:
        print(f"Fatal error: {e}")
        return

    total_samples = len(test_samples)
    if total_samples == 0:
        print("Fatal error: No valid samples found in test file.")
        return

    print(f"Loaded {total_samples} valid samples")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Context length (model/judge): {MODEL_MAX_CONTEXT_TOKENS}/{JUDGE_MODEL_MAX_CONTEXT_TOKENS}")

    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS,
        limit_per_host=MAX_CONCURRENT_REQUESTS,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(process_sample(session, i, s)) for i, s in enumerate(test_samples, 1)]

        final_results: List[Dict[str, Any]] = []
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            try:
                final_results.append(await task)
                if i % 50 == 0:
                    print(f"Progress: {i}/{total_samples}")
            except Exception as e:
                final_results.append(
                    {
                        "sample_index": i,
                        "status": "failed",
                        "error": f"Task processing failed: {str(e)[:200]}",
                    }
                )

    output_file = "~/xxx.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Failed to save results: {e}")
        fallback_file = "/tmp/evaluation_result_fallback.json"
        with open(fallback_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"Fallback saved to: {fallback_file}")

    success_count = sum(1 for r in final_results if r.get("status") == "success")
    failed_count = sum(1 for r in final_results if r.get("status") == "failed")
    success_rate = (success_count / total_samples) * 100 if total_samples else 0.0

    if failed_count:
        print("\nFailure examples (first 5):")
        for r in [x for x in final_results if x.get("status") == "failed"][:5]:
            print(f"- Sample {r.get('sample_index')}: {str(r.get('error', ''))[:200]}")

    print("\nEvaluation completed!")


if __name__ == "__main__":

    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    asyncio.run(main())

