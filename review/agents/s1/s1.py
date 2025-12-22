# agents/s1/s1_agent.py
import json
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Coroutine

from ..base_agent import BaseAgent
from .cheating_detector import CheatingDetector
from .motivation_evaluator import MotivationEvaluator
from .summarizer import Summarizer
from .paper_memory_summarizer import PaperMemorySummarizer
from ..logger import get_logger, get_artifact_logger

logger = get_logger(__name__)
artifact_logger = get_artifact_logger("s1_artifacts")


class S1Agent(BaseAgent):
    def __init__(
        self,
        *,
        model: str,
        reasoning_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            reasoning_model=reasoning_model,
            embedding_model=embedding_model,
            base_url=base_url,
            api_key=api_key,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
        )
        shared_kwargs = dict(
            model=model,
            reasoning_model=reasoning_model,
            embedding_model=embedding_model,
            base_url=base_url,
            api_key=api_key,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
        )
        self.cheating_detector = CheatingDetector(**shared_kwargs)
        self.motivation_evaluator = MotivationEvaluator(**shared_kwargs)
        self.summarizer = Summarizer(**shared_kwargs)
        self.paper_memory_summarizer = PaperMemorySummarizer(**shared_kwargs)
        # PaperStructurer is deprecated in favor of consuming pre-sectioned JSON

    def _parse_sse_chunk(self, sse_string: str) -> str:
        """
        Parse SSE formatted chunk and extract the actual content.
        
        Input format: data: {"object": "chat.completion.chunk", "choices": [{"delta": {"content": "text"}}]}\n\n
        Output: "text"
        """
        try:
            # Remove "data: " prefix and trailing whitespace
            if sse_string.startswith("data: "):
                sse_string = sse_string[6:]
            sse_string = sse_string.strip()
            
            # Skip [DONE] markers
            if sse_string == "[DONE]":
                return ""
            
            # Parse JSON
            data = json.loads(sse_string)
            
            # Extract content from the structured response
            if (isinstance(data, dict) and 
                data.get("choices") and 
                len(data["choices"]) > 0 and
                data["choices"][0].get("delta") and
                data["choices"][0]["delta"].get("content") is not None):
                return data["choices"][0]["delta"]["content"]
            
            return ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.debug(f"Failed to parse SSE chunk: {e}")
            return ""

    def _create_chunk(self, content: str) -> str:
        response_data = {
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {
                    "content": content
                }
            }]
        }
        return f"data: {json.dumps(response_data)}\n\n"

    def _build_sections_from_blocks(self, blocks: List[Dict[str, Any]], enable_mm: bool) -> List[Dict[str, str]]:
        """Group normalized blocks by their `section` field to avoid LLM-based structuring."""

        sections: List[Dict[str, str]] = []
        current_title: Optional[str] = None
        buffer: List[Dict[str, Any]] = []

        def flush():
            nonlocal buffer, current_title
            if not buffer:
                return
            content = self.blocks_to_text(buffer, enable_mm=enable_mm)
            if content.strip():
                sections.append({
                    "title": current_title or "Unknown Section",
                    "content": content,
                })
            buffer = []

        for block in blocks:
            title = (block.get("section") or "").strip() or None
            if current_title is None:
                current_title = title
            elif title is not None and title != current_title:
                flush()
                current_title = title
            buffer.append(block)

        flush()

        if not sections:
            sections.append({
                "title": "Full Paper",
                "content": self.blocks_to_text(blocks, enable_mm=enable_mm),
            })

        return sections

    async def _run_with_progress(self, task_coro, description: str, result_container: List[Any]):
        task = asyncio.create_task(task_coro)
        yield self._create_chunk(f"> {description}...\n> ")
        
        while not task.done():
            done, _ = await asyncio.wait([task], timeout=3.0)
            if not done:
                yield self._create_chunk(".")
        
        yield self._create_chunk("\n")
        result_container.append(task.result())

    async def run(
        self,
        paper_json: Any,
        query: str,
        *,
        enable_mm: bool = False,
        enable_cheating_detection: bool = True,
        enable_motivation: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Execute paper review task with structured JSON input."""
        # ✅ NEW: 简单的落盘工具函数 (改为只记录到 artifact log)
        def _save_text(name: str, text: str):
            # Log the content to artifact logger
            artifact_logger.info(f"\n=== {name} ===\n{text}")

        def _save_json(name: str, obj: Any):
            # Log the content to artifact logger
            artifact_logger.info(f"\n=== {name} ===\n{json.dumps(obj, ensure_ascii=False, indent=2)}")
        blocks = self.prepare_paper_blocks(paper_json)
        raw_pdf_text = self.blocks_to_text(blocks, enable_mm=enable_mm)
        logger.info(f"Normalized paper blocks: {len(blocks)} items; text length={len(raw_pdf_text)}")

        # ✅ NEW: 原始抽取文本也可以落盘，方便 debug
        _save_text("raw_pdf_text", raw_pdf_text)
        _save_json("paper_blocks", blocks)

        # =========================
        # Step 0: 按 section 字段直接聚合章节
        # =========================
        yield self._create_chunk("> (1/4) Aggregating sections from paper JSON...\n")
        sections: List[Dict[str, str]] = self._build_sections_from_blocks(blocks, enable_mm=enable_mm)
        for i, sec in enumerate(sections, start=1):
            logger.info(f"  - Section {i}: {sec.get('title', '')} (len={len(sec.get('content',''))} chars)")
        yield self._create_chunk("\n")

        # ✅ NEW: 保存章节结构
        _save_json("sections", sections)

        # 用规整后的内容拼一个“结构化全文”，供 memory summarizer 使用
        normalized_paper_text = "\n\n".join(
            f"# {sec['title']}\n{sec['content']}" for sec in sections
        )
        logger.info(f"Normalized paper text length: {len(normalized_paper_text)} characters")

        # ✅ NEW: 保存规整后的全文
        _save_text("normalized_paper", normalized_paper_text)

        # =========================
        # Step 1: 构建论文 memory（基于规整后的全文）
        # =========================

        async def collect_paper_memory() -> str:
            logger.info("Starting paper memory summarization...")
            try:
                # Try to consume as async generator first, then fall back to str
                memory_chunks: List[str] = []
                async for chunk in self.paper_memory_summarizer.run(
                    normalized_paper_text, section_titles=[s.get("title", "") for s in sections]
                ):
                    memory_chunks.append(chunk)
                memory = "".join(memory_chunks)
                logger.info(f"Paper memory summarization completed, length={len(memory)}")
                return memory
            except TypeError:
                # If it returns str directly (not async generator)
                try:
                    memory = await self.paper_memory_summarizer.run(
                        normalized_paper_text, section_titles=[s.get("title", "") for s in sections]
                    )
                    logger.info(f"Paper memory summarization completed, length={len(memory)}")
                    return memory
                except Exception as e:
                    logger.error(f"Paper memory summarization failed: {e}")
                    return f"[MEMORY_ERROR] {e}"
            except Exception as e:
                logger.error(f"Paper memory summarization failed: {e}")
                return f"[MEMORY_ERROR] {e}"

        paper_memory_res = []
        async for chunk in self._run_with_progress(collect_paper_memory(), "(2/4) Summarizing paper memory", paper_memory_res):
            yield chunk
        paper_memory = paper_memory_res[0]
        logger.info(f"Paper memory length: {len(paper_memory)} characters")

        # ✅ NEW: 保存 memory
        _save_text("paper_memory", paper_memory)

        # =========================
        # Step 2: 并行跑 cheating + motivation（可开关）
        # =========================

        async def collect_cheating():
            logger.info("Starting section-level critical review via CheatingDetector.run_sectionwise...")
            try:
                result = await self.cheating_detector.run_sectionwise(sections, paper_memory)
                logger.info("CheatingDetector.run_sectionwise completed.")
                return result
            except Exception as e:
                logger.error(f"CheatingDetector.run_sectionwise failed: {e}")
                return f"[CheatingDetector ERROR] {e}"

        async def collect_motivation() -> str:
            logger.info("Starting motivation evaluation (using paper memory)...")
            try:
                # Try to consume as async generator first, then fall back to str
                mot_chunks: List[str] = []
                async for chunk in self.motivation_evaluator.run(paper_memory):
                    mot_chunks.append(chunk)
                mot = "".join(mot_chunks)
                logger.info(f"Motivation evaluation completed, length={len(mot)}")
                return mot
            except TypeError:
                # If it returns str directly (not async generator)
                try:
                    mot = await self.motivation_evaluator.run(paper_memory)
                    logger.info(f"Motivation evaluation completed, length={len(mot)}")
                    return mot
                except Exception as e:
                    logger.error(f"Motivation evaluation failed: {e}")
                    return f"[MOTIVATION_ERROR] {e}"
            except Exception as e:
                logger.error(f"Motivation evaluation failed: {e}")
                return f"[MOTIVATION_ERROR] {e}"

        parallel_tasks: List[tuple[str, asyncio.Future | asyncio.Task | Coroutine[Any, Any, Any]]] = []
        if enable_cheating_detection:
            parallel_tasks.append(("cheating detection", collect_cheating()))
        if enable_motivation:
            parallel_tasks.append(("motivation evaluation", collect_motivation()))

        cheating_detection: Any = "[Cheating detection disabled]"
        motivation_evaluation: Any = "[Motivation evaluation disabled]"

        if parallel_tasks:
            async def run_parallel_tasks():
                coros = [task for _, task in parallel_tasks]
                return await asyncio.gather(*coros)

            stage_desc = " & ".join([name for name, _ in parallel_tasks])
            parallel_res: list[list[Any]] = []
            async for chunk in self._run_with_progress(
                run_parallel_tasks(),
                f"(3/4) Running {stage_desc}",
                parallel_res,
            ):
                yield chunk

            results = parallel_res[0]
            idx = 0
            if enable_cheating_detection:
                cheating_detection = results[idx]
                idx += 1
            if enable_motivation:
                motivation_evaluation = results[idx] if idx < len(results) else motivation_evaluation
        else:
            # 当两者都关闭时，直接进入总结阶段
            yield self._create_chunk(
                "(3/4) Skipping cheating detection and motivation (both disabled).\n"
            )

        # ✅ NEW: 保存 cheating_detection
        if isinstance(cheating_detection, (dict, list)):
            _save_json("cheating_detection", cheating_detection)
        else:
            # 有可能是 str 或错误信息
            _save_text("cheating_detection", str(cheating_detection))

        # ✅ NEW: 保存 motivation 结果
        _save_text("motivation_evaluation", str(motivation_evaluation))

        # =========================
        # Step 3: 把中间结果吐给前端
        # =========================
        # yield self._create_chunk("> Intermediate results generated.\n")

        # =========================
        # Step 4: Summarization 总审稿（这里仍然是流式 Summarizer）
        # =========================
        yield self._create_chunk("> (4/4) Starting final summarization...\n\n---\n\n")

        # ✅ NEW: 把流式总审稿结果也在本地拼起来保存一份
        summary_chunks: List[str] = []

        async for chunk in self.summarizer.run(
            normalized_paper_text,  # 也可以改成 paper_memory，看 Summarizer 的设计
            query,
            str(cheating_detection), # Ensure string
            str(motivation_evaluation), # Ensure string
        ):
            # Parse SSE formatted chunk to extract actual content
            content = self._parse_sse_chunk(chunk)
            summary_chunks.append(content)
            yield chunk

        final_summary = "".join(summary_chunks)
        _save_text("final_summary", final_summary)
        logger.info(f"Final summary saved.")

    async def _run_sync_internal(
        self,
        paper_json: Any,
        query: str,
        *,
        enable_mm: bool = False,
        enable_cheating_detection: bool = True,
        enable_motivation: bool = True,
    ) -> Dict[str, Any]:
        """
        Internal async method to run the full review pipeline and return structured results.
        
        Returns:
            Dict containing:
            - final_assessment: str - The complete final assessment (full text, not chunks)
            - sections: List[Dict] - Structured paper sections with titles and content
            - normalized_paper: str - Restructured full paper text
            - paper_memory: str - Summary of paper key points
            - cheating_detection: dict/str - Cheating detection results by section
            - motivation_evaluation: str - Motivation evaluation results
            - raw_pdf_text: str - Original extracted PDF text
        """
        # Artifact collection functions
        artifacts = {}
        
        def _save_text(name: str, text: str):
            # Log to artifact logger
            artifact_logger.info(f"\n=== {name} ===\n{text}")
            # Store in artifacts dict for downstream persistence
            artifacts[name] = text

        def _save_json(name: str, obj: Any):
            # Log to artifact logger
            artifact_logger.info(f"\n=== {name} ===\n{json.dumps(obj, ensure_ascii=False, indent=2)}")
            # Store in artifacts dict
            artifacts[name] = obj

        # 0) Normalize paper JSON
        blocks = self.prepare_paper_blocks(paper_json)
        raw_pdf_text = self.blocks_to_text(blocks, enable_mm=enable_mm)
        logger.info(f"Normalized paper blocks: {len(blocks)} items; text length={len(raw_pdf_text)}")
        _save_text("raw_pdf_text", raw_pdf_text)
        _save_json("paper_blocks", blocks)

        # Step 0: Structure paper sections directly from normalized blocks
        logger.info("Aggregating sections from paper JSON (no LLM structuring)...")
        sections: List[Dict[str, str]] = self._build_sections_from_blocks(blocks, enable_mm=enable_mm)
        for i, sec in enumerate(sections, start=1):
            title = sec.get("title", "")
            content_len = len(sec.get("content", ""))
            logger.info(f"  - Section {i}: {title} (len={content_len} chars)")

        _save_json("sections", sections)

        # Create normalized paper text
        normalized_paper_text = "\n\n".join(
            f"# {sec['title']}\n{sec['content']}" for sec in sections
        )
        logger.info(f"Normalized paper text length: {len(normalized_paper_text)} characters")
        _save_text("normalized_paper", normalized_paper_text)

        # Step 1: Build paper memory summary
        logger.info("Starting paper memory summarization...")
        try:
            paper_memory_chunks: List[str] = []
            async for chunk in self.paper_memory_summarizer.run(
                normalized_paper_text, section_titles=[s.get("title", "") for s in sections]
            ):
                paper_memory_chunks.append(chunk)
            paper_memory = "".join(paper_memory_chunks)
            logger.info(f"Paper memory summarization completed, length={len(paper_memory)}")
        except TypeError:
            # If paper_memory_summarizer.run returns str directly (not async generator)
            try:
                paper_memory = await self.paper_memory_summarizer.run(
                    normalized_paper_text, section_titles=[s.get("title", "") for s in sections]
                )
                logger.info(f"Paper memory summarization completed, length={len(paper_memory)}")
            except Exception as e:
                logger.error(f"Paper memory summarization failed: {e}")
                paper_memory = f"[MEMORY_ERROR] {e}"
        except Exception as e:
            logger.error(f"Paper memory summarization failed: {e}")
            paper_memory = f"[MEMORY_ERROR] {e}"

        _save_text("paper_memory", paper_memory)

        cheating_detection: Any = "[Cheating detection disabled]"
        motivation_evaluation: Any = "[Motivation evaluation disabled]"

        if enable_cheating_detection:
            logger.info("Starting section-level critical review via CheatingDetector...")
            try:
                cheating_detection = await self.cheating_detector.run_sectionwise(sections, paper_memory)
                logger.info("CheatingDetector.run_sectionwise completed.")
            except Exception as e:
                logger.error(f"CheatingDetector.run_sectionwise failed: {e}")
                cheating_detection = f"[CheatingDetector ERROR] {e}"

        if enable_motivation:
            logger.info("Starting motivation evaluation...")
            try:
                motivation_chunks: List[str] = []
                async for chunk in self.motivation_evaluator.run(paper_memory):
                    motivation_chunks.append(chunk)
                motivation_evaluation = "".join(motivation_chunks)
                logger.info(f"Motivation evaluation completed, length={len(motivation_evaluation)}")
            except TypeError:
                # If motivation_evaluator.run returns str directly (not async generator)
                try:
                    motivation_evaluation = await self.motivation_evaluator.run(paper_memory)
                    logger.info(f"Motivation evaluation completed, length={len(motivation_evaluation)}")
                except Exception as e:
                    logger.error(f"Motivation evaluation failed: {e}")
                    motivation_evaluation = f"[MOTIVATION_ERROR] {e}"
            except Exception as e:
                logger.error(f"Motivation evaluation failed: {e}")
                motivation_evaluation = f"[MOTIVATION_ERROR] {e}"

        # Save cheating detection results
        if isinstance(cheating_detection, (dict, list)):
            _save_json("cheating_detection", cheating_detection)
        else:
            _save_text("cheating_detection", str(cheating_detection))

        # Save motivation evaluation results
        _save_text("motivation_evaluation", str(motivation_evaluation))

        # Step 3: Generate final summary
        logger.info("Starting final summarization...")
        summary_chunks: List[str] = []

        try:
            async for chunk in self.summarizer.run(
                normalized_paper_text,
                query,
                str(cheating_detection),
                str(motivation_evaluation),
            ):
                # Parse SSE formatted chunk to extract actual content
                content = self._parse_sse_chunk(chunk)
                summary_chunks.append(content)
        except Exception as e:
            logger.error(f"Summarizer.run failed: {e}")
            summary_chunks = [f"[SUMMARIZER ERROR] {e}"]

        final_summary = "".join(summary_chunks)
        _save_text("final_summary", final_summary)
        logger.info("Final summary completed.")

        # Return structured results
        return {
            "final_assessment": final_summary,
            "sections": sections,
            "normalized_paper": normalized_paper_text,
            "paper_memory": paper_memory,
            "cheating_detection": cheating_detection,
            "motivation_evaluation": motivation_evaluation,
            "raw_pdf_text": raw_pdf_text,
            "artifacts": artifacts,
        }

    def review_paper(
        self,
        paper_json: Any,
        query: str,
        *,
        enable_mm: bool = False,
        enable_cheating_detection: bool = True,
        enable_motivation: bool = True,
    ) -> Dict[str, Any]:
        """Synchronous wrapper that accepts structured paper JSON input."""

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self._run_sync_internal(
                paper_json,
                query,
                enable_mm=enable_mm,
                enable_cheating_detection=enable_cheating_detection,
                enable_motivation=enable_motivation,
            )
        )

        logger.info("Paper review completed successfully.")
        return result
