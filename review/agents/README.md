# Agents Module

这个模块包含了用于测试 Science Arena Challenge API 的各种 Agent 类。

## 架构设计

### BaseAgent

所有 Agent 的基类，提供通用功能：
- HTTP 客户端管理（使用 `httpx`）
- SSE (Server-Sent Events) 流解析
- 统一的流式输出接口

### 具体 Agent 类

1. **HealthCheckAgent** - 健康检查
   - 测试 `/health` 端点
   - 验证 API 是否正常运行
   - 返回状态码和响应数据

2. **LiteratureReviewAgent** - 文献综述
   - 测试 `/literature_review` 端点
   - 使用标准 LLM 模型进行文献综述
   - **流式输出**：返回异步生成器

3. **PaperQAAgent** - 论文问答
   - 测试 `/paper_qa` 端点
   - 使用推理模型分析 PDF 内容并回答问题
   - 支持 PDF 文件路径或 base64 编码字符串作为输入
   - **流式输出**：返回异步生成器

4. **IdeationAgent** - 研究创意生成
   - 测试 `/ideation` 端点
   - 使用 embedding 模型计算相似度
   - 使用 LLM 生成创新研究想法
   - **流式输出**：返回异步生成器

5. **PaperReviewAgent** - 论文评审
   - 测试 `/paper_review` 端点
   - 对提供的 PDF 论文进行全面评审
   - 支持 PDF 文件路径或 base64 编码字符串作为输入
   - **流式输出**：返回异步生成器

## 使用方法

### 基本用法（流式输出）

```python
from agents import LiteratureReviewAgent

# 创建 Agent 实例
agent = LiteratureReviewAgent(base_url="http://localhost:3000")

# 执行任务并流式接收结果
async for chunk in agent.run(query="What are the key innovations in transformer models?"):
    print(chunk, end="", flush=True)
```

### 使用 PDF 文件（支持路径和 base64）

```python
from agents import PaperQAAgent

# 方式1：传递文件路径（推荐）
agent = PaperQAAgent(base_url="http://localhost:3000")
async for chunk in agent.run(
    query="What is the main contribution of this paper?",
    pdf_content="path/to/paper.pdf"  # 直接传递文件路径
):
    print(chunk, end="", flush=True)

# 方式2：传递 base64 编码的字符串
import base64
with open("paper.pdf", "rb") as f:
    pdf_base64 = base64.b64encode(f.read()).decode('utf-8')

async for chunk in agent.run(
    query="What is the main contribution of this paper?",
    pdf_content=pdf_base64  # 传递 base64 字符串
):
    print(chunk, end="", flush=True)
```

### 健康检查（非流式）

```python
from agents import HealthCheckAgent

agent = HealthCheckAgent(base_url="http://localhost:3000")
result = await agent.run()
print(f"Status: {result['status_code']}")
print(f"Data: {result['data']}")
```

### 自定义超时

```python
# 设置自定义超时时间（秒）
agent = LiteratureReviewAgent(base_url="http://localhost:3000", timeout=180.0)
```

### 自定义 Agent

继承 `BaseAgent` 创建自定义 Agent：

```python
from agents.base_agent import BaseAgent
from typing import AsyncGenerator

class CustomAgent(BaseAgent):
    async def run(self, query: str) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/custom_endpoint",
                json={"query": query}
            ) as response:
                response.raise_for_status()
                # 使用基类的 SSE 解析器
                async for chunk in self._parse_sse_stream(response):
                    yield chunk
```

## 特性

- **解耦设计**: 每个 Agent 独立负责一个 API 端点
- **流式输出**: 所有 Agent（除健康检查外）都返回异步生成器，支持流式处理
- **灵活输入**: PDF 可以作为文件路径或 base64 字符串传入
- **第三方库**: 使用 `httpx` 处理 HTTP 请求和流式响应
- **易于扩展**: 继承 `BaseAgent` 即可创建新的 Agent
- **统一接口**: 所有 Agent 都实现 `run()` 方法
- **错误处理**: 使用 `response.raise_for_status()` 自动处理 HTTP 错误
- **可配置**: 支持自定义 base_url、超时时间等参数

## 依赖

```python
httpx  # 现代异步 HTTP 客户端
```

## 文件结构

```
agents/
├── __init__.py                  # 模块导出
├── base_agent.py               # 基类（SSE 流解析）
├── health_check_agent.py       # 健康检查 Agent
├── literature_review_agent.py  # 文献综述 Agent（流式）
├── paper_qa_agent.py           # 论文问答 Agent（流式）
├── ideation_agent.py           # 创意生成 Agent（流式）
└── paper_review_agent.py       # 论文评审 Agent（流式）
```

## 运行测试

使用 `test_api.py` 运行所有 Agent 测试：

```bash
python test_api.py
```

测试脚本展示了如何：
- 使用流式 Agent 并实时打印输出
- 传递 PDF 文件路径作为参数
- 处理健康检查的非流式响应
