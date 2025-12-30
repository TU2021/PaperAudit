import os
import json
import re
import asyncio
import aiohttp
from typing import Dict, Any, List

import tiktoken

# -------------------------- Dataset --------------------------
TEST_FILE_PATH = ""

# -------------------------- External API (OpenAI-compatible) --------------------------
EXTERNAL_BASE_URL = ""
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "")

MODELS_TO_TEST = [
    "",
]

# -------------------------- Judge (local vLLM) --------------------------
JUDGE_MODEL_NAME = ""
JUDGE_API_KEY = "EMPTY"
JUDGE_BASE_URL = ""
JUDGE_TEMPERATURE = 0.0

# -------------------------- Runtime limits --------------------------
MODEL_MAX_CONTEXT_TOKENS = 8192
JUDGE_MODEL_MAX_CONTEXT_TOKENS = 8192
TOKEN_BUFFER = 400
MAX_INPUT_TRUNCATE_LENGTH = 5500
MAX_COMPLETION_CAP = 4096

MAX_EXTERNAL_INFLIGHT = 40
MAX_JUDGE_INFLIGHT = 64
MODEL_WORKERS_PER_MODEL = 24

RETRY_TIMES = 3
REQUEST_TIMEOUT = 300
RERUN_FAILED_PRED = False

# -------------------------- Outputs --------------------------
OUTPUT_DIR = ""


# -------------------------- Helpers --------------------------
def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def load_test_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSON array and keep only samples containing 'instruction' and 'output'."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Test file must contain a JSON array (list of samples).")

    valid_samples = [s for s in data if isinstance(s, dict) and "instruction" in s and "output" in s]
    if len(valid_samples) < len(data):
        print(f"Warning: Filtered out {len(data) - len(valid_samples)} invalid samples.")
    return valid_samples


def estimate_token_count(text: str) -> int:
    """Estimate token count using cl100k_base; fallback to a rough heuristic on failure."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text) / 3.5)


def truncate_long_input(text: str, max_tokens: int) -> str:
    """Truncate input by tokens (best-effort) and add minimal closing braces to avoid crashes."""
    tok = estimate_token_count(text)
    if tok <= max_tokens:
        return text

    enc = tiktoken.get_encoding("cl100k_base")
    truncated = enc.decode(enc.encode(text)[:max_tokens])

    if truncated.startswith("[") and not truncated.endswith("]"):
        truncated += "]"
    if truncated.startswith("{") and not truncated.endswith("}"):
        truncated += "}"

    print(f"Warning: Truncated input from {tok} to {max_tokens} tokens.")
    return truncated


def clean_prediction_text(text: str) -> str:
    """Remove common wrappers like ```json ... ``` and leading 'json' tokens."""
    if not text:
        return text
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    t = re.sub(r"^\s*json\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


async def append_jsonl(path: str, obj: Dict[str, Any], lock: asyncio.Lock) -> None:
    """Append one JSON object per line to a JSONL file (guarded by a lock to avoid interleaving)."""
    line = json.dumps(obj, ensure_ascii=False)
    async with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def load_jsonl_state(path: str) -> Dict[int, Dict[str, Any]]:
    """
    Resume state from JSONL logs. Per sample_index, track:
      pred_done/pred_success/prediction and judge_done/judge_success.
    """
    state: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return state

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue

            idx = rec.get("sample_index")
            if not isinstance(idx, int):
                continue

            st = state.setdefault(
                idx,
                {
                    "pred_done": False,
                    "pred_success": False,
                    "prediction": None,
                    "judge_done": False,
                    "judge_success": False,
                },
            )

            rtype = rec.get("record_type")
            status = rec.get("status")

            if rtype == "pred":
                st["pred_done"] = True
                st["pred_success"] = (status == "success")
                pred = rec.get("prediction")
                if isinstance(pred, str) and pred.strip():
                    st["prediction"] = pred

            elif rtype == "judge":
                st["judge_done"] = True
                st["judge_success"] = (status == "success")
                pred = rec.get("prediction")
                if isinstance(pred, str) and pred.strip():
                    st["prediction"] = pred

    return state


def merge_jsonl_to_json_array(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Merge JSONL into a JSON array:
      - One final record per sample_index
      - Prefer the latest judge record; fallback to the latest pred record
    """
    if not os.path.exists(jsonl_path):
        return []

    last_pred: Dict[int, Dict[str, Any]] = {}
    last_judge: Dict[int, Dict[str, Any]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue

            idx = rec.get("sample_index")
            if not isinstance(idx, int):
                continue

            if rec.get("record_type") == "pred":
                last_pred[idx] = rec
            elif rec.get("record_type") == "judge":
                last_judge[idx] = rec

    out: List[Dict[str, Any]] = []
    for idx in sorted(set(last_pred.keys()) | set(last_judge.keys())):
        rec = last_judge.get(idx) or last_pred.get(idx)
        if not rec:
            continue
        r = dict(rec)
        r.pop("record_type", None)
        out.append(r)
    return out


# -------------------------- API calls --------------------------
async def async_call_model_api(
    session: aiohttp.ClientSession,
    prompt: str,
    model_name: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_context_tokens: int,
    external_sema: asyncio.Semaphore,
    judge_sema: asyncio.Semaphore,
    retry: int = 0,
) -> str:
    """OpenAI-compatible /chat/completions call with bounded concurrency and retries."""
    prompt = truncate_long_input(prompt, MAX_INPUT_TRUNCATE_LENGTH)
    in_tok = estimate_token_count(prompt)

    max_completion = max_context_tokens - in_tok - TOKEN_BUFFER
    max_completion = max(100, min(max_completion, MAX_COMPLETION_CAP))

    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_completion,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    sema = external_sema if base_url.startswith(EXTERNAL_BASE_URL) else judge_sema

    try:
        async with sema:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == 429:
                    if retry < RETRY_TIMES:
                        sleep_time = 2**retry
                        print(f"Warning: 429 rate limited (retry {retry + 1}/{RETRY_TIMES}), sleep {sleep_time}s")
                        await asyncio.sleep(sleep_time)
                        return await async_call_model_api(
                            session,
                            prompt,
                            model_name,
                            api_key,
                            base_url,
                            temperature,
                            max_context_tokens,
                            external_sema,
                            judge_sema,
                            retry + 1,
                        )
                    raise RuntimeError("Rate limit exceeded (429).")

                if resp.status != 200:
                    txt = await resp.text()
                    if resp.status == 503 and "No available channels" in txt:
                        raise RuntimeError(f"Unretryable 503: {txt[:500]}")
                    raise RuntimeError(f"API request failed (status {resp.status}): {txt[:800]}")

                result = await resp.json()
                if not result.get("choices"):
                    raise RuntimeError("Empty response from model API.")

                return result["choices"][0]["message"]["content"].strip()

    except asyncio.TimeoutError:
        if retry < RETRY_TIMES:
            print(f"Warning: timeout (retry {retry + 1}/{RETRY_TIMES})")
            await asyncio.sleep(2)
            return await async_call_model_api(
                session,
                prompt,
                model_name,
                api_key,
                base_url,
                temperature,
                max_context_tokens,
                external_sema,
                judge_sema,
                retry + 1,
            )
        raise

    except Exception as e:
        if retry < RETRY_TIMES and "Unretryable 503" not in str(e):
            print(f"Warning: API error (retry {retry + 1}/{RETRY_TIMES}): {str(e)[:240]}")
            await asyncio.sleep(1)
            return await async_call_model_api(
                session,
                prompt,
                model_name,
                api_key,
                base_url,
                temperature,
                max_context_tokens,
                external_sema,
                judge_sema,
                retry + 1,
            )
        raise


def construct_judge_prompt(ground_truth: List[Dict[str, Any]], prediction: str) -> str:
    """Build a JSON-only evaluation prompt for the judge model."""
    return f"""
You are a strict but fair professional evaluator for academic paper error detection tasks.
Compare the ground truth error annotations with the model's prediction and output valid JSON.

## Output Requirements:
- Output ONLY valid JSON (no extra text, no markdown)
- Must include keys: "matches", "overall_match_score"
- matches: list of objects with keys: gt_index, matched, matched_pred_indices, rationale

## Ground Truth:
{json.dumps(ground_truth, ensure_ascii=False, indent=2)}

## Prediction:
{prediction}

## Example Output:
{{
  "matches": [
    {{
      "gt_index": 0,
      "matched": true,
      "matched_pred_indices": [0],
      "rationale": "..."
    }}
  ],
  "overall_match_score": 10
}}
""".strip()


async def async_judge_output(
    session: aiohttp.ClientSession,
    ground_truth: List[Dict[str, Any]],
    prediction: str,
    external_sema: asyncio.Semaphore,
    judge_sema: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Call judge model and parse JSON object containing 'matches' and 'overall_match_score'."""
    judge_prompt = construct_judge_prompt(ground_truth, prediction)
    raw = await async_call_model_api(
        session=session,
        prompt=judge_prompt,
        model_name=JUDGE_MODEL_NAME,
        api_key=JUDGE_API_KEY,
        base_url=JUDGE_BASE_URL,
        temperature=JUDGE_TEMPERATURE,
        max_context_tokens=JUDGE_MODEL_MAX_CONTEXT_TOKENS,
        external_sema=external_sema,
        judge_sema=judge_sema,
    )

    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError(f"No JSON found in judge output. Preview: {raw[:400]}")

    out = json.loads(m.group())
    if "matches" not in out or "overall_match_score" not in out:
        raise ValueError(f"Judge output missing keys: {list(out.keys())}")
    return out


# -------------------------- Per-sample pipeline: pred -> JSONL -> judge -> JSONL --------------------------
async def run_one_sample_for_one_model(
    session: aiohttp.ClientSession,
    model: str,
    sample_idx: int,
    instruction: str,
    ground_truth: List[Dict[str, Any]],
    jsonl_path: str,
    lock: asyncio.Lock,
    state: Dict[int, Dict[str, Any]],
    external_sema: asyncio.Semaphore,
    judge_sema: asyncio.Semaphore,
) -> None:
    st = state.get(
        sample_idx,
        {
            "pred_done": False,
            "pred_success": False,
            "prediction": None,
            "judge_done": False,
            "judge_success": False,
        },
    )

    # Cost-saving rule: if pred succeeded before, do not call external API again.
    need_pred = True
    if st["pred_done"]:
        if st["pred_success"]:
            need_pred = False
        else:
            need_pred = bool(RERUN_FAILED_PRED)

    prediction = st.get("prediction") or ""

    if need_pred:
        wrapped_instruction = (
            "Return ONLY a valid JSON array. No extra text, no markdown. "
            "Keep it concise (<= 10 items). Stop immediately after the closing bracket ']'.\n\n"
            + instruction
        )
        try:
            pred_raw = await async_call_model_api(
                session=session,
                prompt=wrapped_instruction,
                model_name=model,
                api_key=EXTERNAL_API_KEY,
                base_url=EXTERNAL_BASE_URL,
                temperature=0.0,
                max_context_tokens=MODEL_MAX_CONTEXT_TOKENS,
                external_sema=external_sema,
                judge_sema=judge_sema,
            )
            prediction = clean_prediction_text(pred_raw)

            await append_jsonl(
                jsonl_path,
                {
                    "record_type": "pred",
                    "sample_index": sample_idx,
                    "model": model,
                    "status": "success",
                    "instruction": instruction[:300] + "..." if len(instruction) > 300 else instruction,
                    "prediction": prediction,
                },
                lock,
            )

            st["pred_done"] = True
            st["pred_success"] = True
            st["prediction"] = prediction

        except Exception as e:
            await append_jsonl(
                jsonl_path,
                {
                    "record_type": "pred",
                    "sample_index": sample_idx,
                    "model": model,
                    "status": "failed",
                    "error": str(e)[:800],
                    "instruction": instruction[:300] + "..." if len(instruction) > 300 else instruction,
                    "prediction": "",
                },
                lock,
            )
            st["pred_done"] = True
            st["pred_success"] = False
            st["prediction"] = ""

    state[sample_idx] = st

    # Do not re-run judge if done (resume-friendly).
    if st.get("judge_done", False):
        return

    prediction = st.get("prediction") or ""
    if not prediction.strip():
        await append_jsonl(
            jsonl_path,
            {
                "record_type": "judge",
                "sample_index": sample_idx,
                "model": model,
                "status": "failed",
                "error": "Skip judge because prediction is empty/failed",
                "prediction": prediction,
            },
            lock,
        )
        st["judge_done"] = True
        st["judge_success"] = False
        state[sample_idx] = st
        return

    try:
        judge_result = await async_judge_output(session, ground_truth, prediction, external_sema, judge_sema)
        await append_jsonl(
            jsonl_path,
            {
                "record_type": "judge",
                "sample_index": sample_idx,
                "model": model,
                "status": "success",
                "prediction": prediction,
                "judge_result": judge_result,
                "ground_truth_sample": ground_truth[:1],
            },
            lock,
        )
        st["judge_done"] = True
        st["judge_success"] = True

    except Exception as e:
        await append_jsonl(
            jsonl_path,
            {
                "record_type": "judge",
                "sample_index": sample_idx,
                "model": model,
                "status": "failed",
                "prediction": prediction,
                "error": f"Judge failed: {str(e)[:800]}",
                "ground_truth_sample": ground_truth[:1],
            },
            lock,
        )
        st["judge_done"] = True
        st["judge_success"] = False

    state[sample_idx] = st


# -------------------------- Main --------------------------
async def main() -> None:
    if not EXTERNAL_API_KEY:
        raise RuntimeError(
            "Missing EXTERNAL_API_KEY. Set it via environment variable, e.g.:\n"
            "  export EXTERNAL_API_KEY='your_key'\n"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_samples = load_test_file(TEST_FILE_PATH)
    total = len(test_samples)
    if total == 0:
        print("No samples found.")
        return

    jsonl_paths = {m: os.path.join(OUTPUT_DIR, f"{safe_filename(m)}.jsonl") for m in MODELS_TO_TEST}
    json_paths = {m: os.path.join(OUTPUT_DIR, f"{safe_filename(m)}.json") for m in MODELS_TO_TEST}
    locks = {m: asyncio.Lock() for m in MODELS_TO_TEST}
    states = {m: load_jsonl_state(jsonl_paths[m]) for m in MODELS_TO_TEST}

    def is_done(model: str, idx: int) -> bool:
        st = states[model].get(idx)
        if not st or not st.get("pred_done", False):
            return False
        if st.get("pred_success", False):
            return bool(st.get("judge_done", False))
        return (not RERUN_FAILED_PRED) and bool(st.get("judge_done", False))

    queues = {m: asyncio.Queue() for m in MODELS_TO_TEST}
    todo_by_model = {m: 0 for m in MODELS_TO_TEST}

    for m in MODELS_TO_TEST:
        for idx in range(1, total + 1):
            if is_done(m, idx):
                continue
            queues[m].put_nowait(idx)
            todo_by_model[m] += 1

    print("=" * 90)
    print("Multi-model evaluation with JSONL incremental saving + resume")
    print(f"Total samples: {total}")
    print(f"Models: {MODELS_TO_TEST}")
    print(f"Workers per model: {MODEL_WORKERS_PER_MODEL}")
    print(f"External inflight: {MAX_EXTERNAL_INFLIGHT} | Judge inflight: {MAX_JUDGE_INFLIGHT}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("Todo per model:", todo_by_model)
    print("=" * 90)

    external_sema = asyncio.Semaphore(MAX_EXTERNAL_INFLIGHT)
    judge_sema = asyncio.Semaphore(MAX_JUDGE_INFLIGHT)

    connector = aiohttp.TCPConnector(
        limit=MAX_EXTERNAL_INFLIGHT + MAX_JUDGE_INFLIGHT,
        limit_per_host=MAX_EXTERNAL_INFLIGHT + MAX_JUDGE_INFLIGHT,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    async with aiohttp.ClientSession(connector=connector) as session:

        async def model_worker(model: str, wid: int) -> None:
            processed = 0
            q = queues[model]
            jsonl_path = jsonl_paths[model]
            lock = locks[model]
            state = states[model]

            while True:
                sample_idx = await q.get()
                if sample_idx is None:
                    q.task_done()
                    break

                try:
                    sample = test_samples[sample_idx - 1]
                    instruction = (sample.get("instruction") or "").strip()
                    ref = (sample.get("output") or "").strip()

                    # Always log invalid samples to avoid re-processing on resume.
                    if not instruction or not ref:
                        await append_jsonl(
                            jsonl_path,
                            {
                                "record_type": "pred",
                                "sample_index": sample_idx,
                                "model": model,
                                "status": "failed",
                                "error": "Invalid sample: missing instruction/output",
                                "prediction": "",
                            },
                            lock,
                        )
                        await append_jsonl(
                            jsonl_path,
                            {
                                "record_type": "judge",
                                "sample_index": sample_idx,
                                "model": model,
                                "status": "failed",
                                "error": "Skip judge: invalid sample",
                                "prediction": "",
                            },
                            lock,
                        )
                        state[sample_idx] = {
                            "pred_done": True,
                            "pred_success": False,
                            "prediction": "",
                            "judge_done": True,
                            "judge_success": False,
                        }
                        continue

                    try:
                        ground_truth = json.loads(ref)
                        if not isinstance(ground_truth, list):
                            raise ValueError("Ground truth must be a JSON array (list).")
                    except Exception as e:
                        await append_jsonl(
                            jsonl_path,
                            {
                                "record_type": "pred",
                                "sample_index": sample_idx,
                                "model": model,
                                "status": "failed",
                                "error": f"Invalid ground truth: {str(e)[:300]}",
                                "prediction": "",
                            },
                            lock,
                        )
                        await append_jsonl(
                            jsonl_path,
                            {
                                "record_type": "judge",
                                "sample_index": sample_idx,
                                "model": model,
                                "status": "failed",
                                "error": "Skip judge: invalid ground truth",
                                "prediction": "",
                            },
                            lock,
                        )
                        state[sample_idx] = {
                            "pred_done": True,
                            "pred_success": False,
                            "prediction": "",
                            "judge_done": True,
                            "judge_success": False,
                        }
                        continue

                    await run_one_sample_for_one_model(
                        session=session,
                        model=model,
                        sample_idx=sample_idx,
                        instruction=instruction,
                        ground_truth=ground_truth,
                        jsonl_path=jsonl_path,
                        lock=lock,
                        state=state,
                        external_sema=external_sema,
                        judge_sema=judge_sema,
                    )

                    processed += 1
                    if processed % 50 == 0:
                        print(f"[{model} worker{wid}] processed={processed}, latest={sample_idx}")

                finally:
                    q.task_done()

        workers = []
        for m in MODELS_TO_TEST:
            for wid in range(MODEL_WORKERS_PER_MODEL):
                workers.append(asyncio.create_task(model_worker(m, wid)))

        for m in MODELS_TO_TEST:
            await queues[m].join()

        for m in MODELS_TO_TEST:
            for _ in range(MODEL_WORKERS_PER_MODEL):
                queues[m].put_nowait(None)

        await asyncio.gather(*workers, return_exceptions=True)

    print("\nMerging JSONL -> JSON arrays ...")
    for m in MODELS_TO_TEST:
        arr = merge_jsonl_to_json_array(jsonl_paths[m])
        with open(json_paths[m], "w", encoding="utf-8") as f:
            json.dump(arr, f, ensure_ascii=False, indent=2)
        print(f"Saved merged JSON: {json_paths[m]} (records={len(arr)})")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
