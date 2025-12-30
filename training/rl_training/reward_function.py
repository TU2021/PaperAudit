import json
import hashlib
import asyncio
import re
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
import httpx

# ==================== 全局配置区域 ====================

# -------------------- 模型 & API 配置 --------------------
JUDGE_MODEL_NAME = "/mnt/parallel_ssd/home/zdhs0124/AI4S_review/Reward_Model/Qwen3-80B/"
JUDGE_API_KEY = "EMPTY"
JUDGE_BASE_URL = "http://10.1.100.72:8001/v1"
HTTP_CONNECT_TIMEOUT = 30.0
HTTP_READ_TIMEOUT = 180.0
HTTP_WRITE_TIMEOUT = 30.0
HTTP_TOTAL_TIMEOUT = 180.0
HTTP_MAX_CONNECTIONS = 100
HTTP_MAX_KEEPALIVE_CONNECTIONS = 20
HTTP_KEEPALIVE_EXPIRY = 30.0

REWARD_CACHE_MAX_SIZE = 10000
CACHE_KEY_SEPARATOR = "|||"

PRECISION_WEIGHT = 0.6
CONCISENESS_WEIGHT = 0.4
PENALTY_INVALID_FORMAT = -0.5
PENALTY_EMPTY_OUTPUT = -0.8
PENALTY_INVALID_GROUND_TRUTH = -0.3
PENALTY_MODEL_CALL_FAILED = 0.0
JUDGE_MODEL_TEMPERATURE = 0.0
JUDGE_MODEL_MAX_TOKENS = 150

# -------------------- 裁判提示词模板 --------------------
REWARD_PROMPT = """
You are a rigorous academic paper reviewer specializing in error detection. Evaluate the model's output based on the following criteria:

【Task Background】
The model must identify scientific errors, methodological flaws, or academic integrity issues in the paper text. Output format: JSON array with each error containing error_location, error_type, and error_explanation.

【Evaluation Criteria】
1. Precision (weight: 0.6):
   - Semantic match with ground truth: If the model's detected errors match ground truth in terms of:
     * error_type (correct classification)
     * error_explanation (reasonable and accurate)
     * error_location (overlaps with ground truth text, at least 50% word overlap or same section)
   Then precision score = 1.0 per match.
   - False positive errors (non-existent errors in ground truth): Deduct 0.2 points per false positive (minimum score 0.0).
   - Duplicate errors (same error reported multiple times): Deduct 0.4 points per duplicate (minimum score 0.0).
   - Missing errors: No penalty (prioritize semantic accuracy of detected errors over completeness).

2. Conciseness (weight: 0.4):
   - No redundant information (only core error detection results, no extra irrelevant content): 1.0 points.
   - Minor redundancy (few repetitive phrases or trivial extra remarks): 0.7 points.
   - Excessive redundant nonsense (large sections of irrelevant content/repetitive explanations): 0.3 points.

【Input Information】
- Ground Truth Errors (reference): {ground_truth}
- Model Output (to evaluate): {model_output}

【Output Requirement】
Return ONLY a JSON object with precision and conciseness (keep 2 decimal places), no extra content:
{{"precision": 0.00, "conciseness": 0.00}}
"""

# ==================== 客户端 & 缓存初始化 ====================
http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=HTTP_MAX_CONNECTIONS,
        max_keepalive_connections=HTTP_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=HTTP_KEEPALIVE_EXPIRY,
    ),
    timeout=httpx.Timeout(
        HTTP_TOTAL_TIMEOUT,
        connect=HTTP_CONNECT_TIMEOUT,
        read=HTTP_READ_TIMEOUT,
        write=HTTP_WRITE_TIMEOUT,
    ),
    follow_redirects=True,
)

client = AsyncOpenAI(
    api_key=JUDGE_API_KEY,
    base_url=JUDGE_BASE_URL,
    timeout=HTTP_TOTAL_TIMEOUT,
    http_client=http_client,
)

reward_cache: Dict[str, float] = {}

# ==================== 工具函数 ====================
def _generate_cache_key(model_output: str, ground_truth: str) -> str:
    """根据模型输出和 ground truth 生成缓存 key"""
    combined = f"{model_output}{CACHE_KEY_SEPARATOR}{ground_truth}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()

async def _call_judge_model(model_output: str, ground_truth: str) -> dict:
    """
    调用裁判模型 (Qwen3-80B) 进行打分
    model_output: 被训练模型（Qwen3-8B）生成的 JSON 数组字符串（清洗后）
    ground_truth: 标准答案（JSON 串）
    """
    prompt = REWARD_PROMPT.format(
        ground_truth=ground_truth,
        model_output=model_output,
    )

    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = await client.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=messages,
            temperature=JUDGE_MODEL_TEMPERATURE,
            max_tokens=JUDGE_MODEL_MAX_TOKENS,
        )
    except Exception as e:
        print(f"Error calling Qwen3-80B: {str(e)}")
        raise

    response_text = response.choices[0].message.content.strip()

    # 清洗 Markdown 代码块
    if "```" in response_text:
        response_text = re.sub(r"```json?\s*", "", response_text)
        response_text = response_text.replace("```", "").strip()

    # 容错解析分数
    try:
        scores = json.loads(response_text)
        precision = max(0.0, min(1.0, float(scores.get("precision", 0.0))))
        conciseness = max(0.0, min(1.0, float(scores.get("conciseness", 0.0))))
    except Exception as e:
        print(f"解析裁判模型输出失败: {str(e)}, 原始输出: {response_text}")
        raise

    return {"precision": precision, "conciseness": conciseness}

# ==================== 主奖励函数（异步） ====================
async def compute_score(
    solution_str: str = None,
    ground_truth: Any = None,
    **kwargs,  # 兼容其他调用方式的透传参数
) -> float:
    """
    异步计算奖励分数（给 RL 使用）

    参数：
    - solution_str: 被训练模型（Qwen3-8B）输出的字符串
    - ground_truth: 可以是 list/dict（包含ground_truth键）
    - kwargs: 兼容其他调用方式的透传参数
    """

    # ==================== 1. 提取模型输出 ====================
    if solution_str and solution_str.strip():
        model_output = solution_str.strip()
    elif "messages" in kwargs and len(kwargs["messages"]) > 0:
        model_output = kwargs["messages"][-1].get("content", "").strip()
    else:
        # 没有任何输出可评估
        return PENALTY_INVALID_FORMAT

    # 去掉 <think> ... </think> 内容
    cleaned = model_output
    if "<think>" in cleaned or "</think>" in cleaned:
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # ==================== 2. 提取 ground_truth ====================
    if isinstance(ground_truth, list):
        actual_gt = ground_truth
    elif isinstance(ground_truth, dict) and "ground_truth" in ground_truth:
        actual_gt = ground_truth["ground_truth"]
    # 新增：如果是字符串，尝试解析为 JSON list
    elif isinstance(ground_truth, str) and ground_truth.strip():
        try:
            actual_gt = json.loads(ground_truth.strip())  # 字符串转 list
        except json.JSONDecodeError as e:
            return PENALTY_INVALID_GROUND_TRUTH
    else:
        # ground_truth 结构不合法
        return PENALTY_INVALID_GROUND_TRUTH

    gt_str = json.dumps(actual_gt, ensure_ascii=False, indent=2)

    # ==================== 3. 检查缓存 ====================
    cache_key = _generate_cache_key(cleaned, gt_str)
    if cache_key in reward_cache:
        print(f"[DEBUG] 缓存命中，直接返回缓存结果")
        return reward_cache[cache_key]

    # ==================== 4. 调用裁判模型 ====================
    try:
        scores = await _call_judge_model(cleaned, gt_str)

        precision = scores.get("precision", 0.0)
        conciseness = scores.get("conciseness", 0.0)
        total_reward = PRECISION_WEIGHT * precision + CONCISENESS_WEIGHT * conciseness

        print(
            f"[DEBUG] ✅ 裁判评分 - precision: {precision:.2f}, "
            f"conciseness: {conciseness:.2f}, 总分: {total_reward:.2f}"
        )

        # ==================== 5. 更新缓存 ====================
        if len(reward_cache) >= REWARD_CACHE_MAX_SIZE:
            oldest_key = next(iter(reward_cache.keys()))
            del reward_cache[oldest_key]
        reward_cache[cache_key] = total_reward

        return float(total_reward)

    except Exception:
        import traceback
        traceback.print_exc()
        return PENALTY_MODEL_CALL_FAILED
