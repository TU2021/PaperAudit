#!/usr/bin/env python3
"""
Simple test script for API endpoints.

Fixes:
 - /health 原代码使用 POST 请求且同步阻塞，已改为异步 GET。
 - 增加连接失败与异常的明确输出。
 - 如果缺少测试 PDF，则跳过 /paper_review 用例并给出提示。
"""
import base64
import httpx
import asyncio
import os
import glob

BASE_URL = "http://localhost:3000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

async def test_health():
    """Test /health endpoint (GET, non-stream)"""
    print_section("0. Testing /health")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(f"{BASE_URL}/health")
            print(f"HTTP Status: {response.status_code}")
            try:
                print(f"Response JSON: {response.json()}")
            except Exception:
                print(f"Raw Response Text: {response.text}")
    except httpx.ConnectError as e:
        print(f"Connection error: {e}. 服务器可能尚未启动或端口不一致 (期望 {BASE_URL}).")
    except Exception as e:
        print(f"Unexpected error during /health: {e}")

async def test_paper_review(pdf_path):
    """Test /paper_review endpoint (streaming)"""
    print_section(f"4. Testing /paper_review for {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"PDF 文件 '{pdf_path}' 未找到，跳过该测试。请放置文件后再运行。")
        return

    try:
        with open(pdf_path, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"读取 PDF 失败: {e}")
        return

    try:
        async with httpx.AsyncClient(timeout=1200.0) as client:
            async with client.stream(
                "POST",
                f"{BASE_URL}/paper_review",
                json={
                    "query": "请扮演一个学术同行评审，提供一个关于这篇论文的详细评审。",
                    "pdf_content": pdf_base64
                }
            ) as response:
                print(f"HTTP Status: {response.status_code}")
                response.raise_for_status()
                print("Streaming Response:")
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    content = line[6:]
                    if content == "[DONE]":
                        print("\n[Stream completed]")
                        break
                    else:
                        import json
                        try:
                            data = json.loads(content)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    print(delta["content"], end="", flush=True)
                        except json.JSONDecodeError:
                            # 非 JSON 行忽略
                            pass
        print("\n")
    except httpx.ConnectError as e:
        print(f"连接失败: {e}. 服务器可能未启动。")
    except httpx.HTTPStatusError as e:
        print(f"HTTP 错误: {e.response.status_code} {e.response.text}")
    except Exception as e:
        print(f"发生未知错误: {e}")
        
    try:
        async with httpx.AsyncClient(timeout=1200.0) as client:
            async with client.stream(
                "POST",
                f"{BASE_URL}/paper_review",
                json={
                    "query": "Please act as an academic reviewer and provide a comprehensive peer review of the paper.",
                    "pdf_content": pdf_base64
                }
            ) as response:
                print(f"HTTP Status: {response.status_code}")
                response.raise_for_status()
                print("Streaming Response:")
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    content = line[6:]
                    if content == "[DONE]":
                        print("\n[Stream completed]")
                        break
                    else:
                        import json
                        try:
                            data = json.loads(content)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    print(delta["content"], end="", flush=True)
                        except json.JSONDecodeError:
                            # 非 JSON 行忽略
                            pass
        print("\n")
    except httpx.ConnectError as e:
        print(f"连接失败: {e}. 服务器可能未启动。")
    except httpx.HTTPStatusError as e:
        print(f"HTTP 错误: {e.response.status_code} {e.response.text}")
    except Exception as e:
        print(f"发生未知错误: {e}")


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("  Science Arena Challenge - API Test Suite")
    print("=" * 80)

    try:
        await test_health()
        
        pdf_files = sorted(glob.glob("data/*.pdf"))
        if not pdf_files:
            print("No PDF files found in data/ directory.")
        
        for pdf_file in pdf_files:
            await test_paper_review(pdf_file)

        print_section("All tests completed!")

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
