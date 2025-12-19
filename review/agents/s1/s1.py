# agents/s1/s1_agent.py
import json
import asyncio
import base64
from typing import AsyncGenerator, List, Dict, Any
from pathlib import Path

from ..base_agent import BaseAgent
from .cheating_detector import CheatingDetector
from .motivation_evaluator import MotivationEvaluator
from .summarizer import Summarizer
from .paper_memory_summarizer import PaperMemorySummarizer
from .paper_structurer import PaperStructurer
from ..logger import get_logger, get_artifact_logger

logger = get_logger(__name__)
artifact_logger = get_artifact_logger("s1_artifacts")


class S1Agent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.cheating_detector = CheatingDetector()
        self.motivation_evaluator = MotivationEvaluator()
        self.summarizer = Summarizer()
        self.paper_memory_summarizer = PaperMemorySummarizer()
        self.paper_structurer = PaperStructurer()

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

    async def _run_with_progress(self, task_coro, description: str, result_container: List[Any]):
        task = asyncio.create_task(task_coro)
        yield self._create_chunk(f"> {description}...\n> ")
        
        while not task.done():
            done, _ = await asyncio.wait([task], timeout=3.0)
            if not done:
                yield self._create_chunk(".")
        
        yield self._create_chunk("\n")
        result_container.append(task.result())

    async def run(self, pdf_content: str, query: str) -> AsyncGenerator[str, None]:
        """
        Execute paper review task with streaming response
        """
        # ✅ NEW: 简单的落盘工具函数 (改为只记录到 artifact log)
        def _save_text(name: str, text: str):
            # Log the content to artifact logger
            artifact_logger.info(f"\n=== {name} ===\n{text}")

        def _save_json(name: str, obj: Any):
            # Log the content to artifact logger
            artifact_logger.info(f"\n=== {name} ===\n{json.dumps(obj, ensure_ascii=False, indent=2)}")
        # 0) base64 → 原始全文文本
        raw_pdf_text = self.extract_pdf_text_from_base64(pdf_content)
        logger.info(f"Extracted raw PDF text length: {len(raw_pdf_text)} characters")

        # ✅ NEW: 原始抽取文本也可以落盘，方便 debug
        _save_text("raw_pdf_text", raw_pdf_text)

        # =========================
        # Step 0: 先用 LLM 规整章节结构
        # =========================
        # yield json.dumps({
        #     "status": "processing",
        #     "stage": "paper_structuring",
        # }) + "\n"

        async def collect_structured_sections() -> List[Dict[str, str]]:
            logger.info("Starting PaperStructurer.build_structure...")
            try:
                sections_: List[Dict[str, str]] = await self.paper_structurer.run(raw_pdf_text)
                logger.info(f"PaperStructurer returned {len(sections_)} sections")
                for i, sec in enumerate(sections_, start=1):
                    title = sec.get("title", "")
                    content_len = len(sec.get("content", ""))
                    logger.info(f"  - Section {i}: {title} (len={content_len} chars)")
                return sections_
            except Exception as e:
                logger.error(f"PaperStructurer.build_structure failed: {e}")
                return [{"title": "Full Paper (fallback from S1Agent)", "content": raw_pdf_text}]

        sections_res = []
        async for chunk in self._run_with_progress(collect_structured_sections(), "(1/4) Structuring paper sections", sections_res):
            yield chunk
        sections: List[Dict[str, str]] = sections_res[0]

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
                async for chunk in self.paper_memory_summarizer.run(raw_pdf_text):
                    memory_chunks.append(chunk)
                memory = "".join(memory_chunks)
                logger.info(f"Paper memory summarization completed, length={len(memory)}")
                return memory
            except TypeError:
                # If it returns str directly (not async generator)
                try:
                    memory = await self.paper_memory_summarizer.run(raw_pdf_text)
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
        # Step 2: 并行跑 cheating + motivation
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

        async def run_parallel_tasks():
            return await asyncio.gather(
                collect_cheating(),
                collect_motivation(),
            )

        parallel_res = []
        async for chunk in self._run_with_progress(
            run_parallel_tasks(),
            "(3/4) Running cheating detection and motivation evaluation",
            parallel_res
        ):
            yield chunk
        cheating_detection, motivation_evaluation = parallel_res[0]

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

    async def _run_sync_internal(self, pdf_content: str, query: str) -> Dict[str, Any]:
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
            # Store in artifacts dict
            if name not in ['raw_pdf_text', 'normalized_paper', 'paper_memory', 'motivation_evaluation']:
                artifacts[name] = text

        def _save_json(name: str, obj: Any):
            # Log to artifact logger
            artifact_logger.info(f"\n=== {name} ===\n{json.dumps(obj, ensure_ascii=False, indent=2)}")
            # Store in artifacts dict
            artifacts[name] = obj

        # 0) Extract PDF text from base64
        raw_pdf_text = self.extract_pdf_text_from_base64(pdf_content)
        logger.info(f"Extracted raw PDF text length: {len(raw_pdf_text)} characters")
        _save_text("raw_pdf_text", raw_pdf_text)

        # Step 0: Structure paper sections using LLM
        logger.info("Starting PaperStructurer.build_structure...")
        try:
            sections: List[Dict[str, str]] = await self.paper_structurer.run(raw_pdf_text)
            logger.info(f"PaperStructurer returned {len(sections)} sections")
            for i, sec in enumerate(sections, start=1):
                title = sec.get("title", "")
                content_len = len(sec.get("content", ""))
                logger.info(f"  - Section {i}: {title} (len={content_len} chars)")
        except Exception as e:
            logger.error(f"PaperStructurer.build_structure failed: {e}")
            sections = [{"title": "Full Paper (fallback)", "content": raw_pdf_text}]

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
            async for chunk in self.paper_memory_summarizer.run(raw_pdf_text):
                paper_memory_chunks.append(chunk)
            paper_memory = "".join(paper_memory_chunks)
            logger.info(f"Paper memory summarization completed, length={len(paper_memory)}")
        except TypeError:
            # If paper_memory_summarizer.run returns str directly (not async generator)
            try:
                paper_memory = await self.paper_memory_summarizer.run(raw_pdf_text)
                logger.info(f"Paper memory summarization completed, length={len(paper_memory)}")
            except Exception as e:
                logger.error(f"Paper memory summarization failed: {e}")
                paper_memory = f"[MEMORY_ERROR] {e}"
        except Exception as e:
            logger.error(f"Paper memory summarization failed: {e}")
            paper_memory = f"[MEMORY_ERROR] {e}"

        _save_text("paper_memory", paper_memory)

        # Step 2: Run cheating detection and motivation evaluation in parallel
        logger.info("Starting section-level critical review via CheatingDetector...")
        try:
            cheating_detection = await self.cheating_detector.run_sectionwise(sections, paper_memory)
            logger.info("CheatingDetector.run_sectionwise completed.")
        except Exception as e:
            logger.error(f"CheatingDetector.run_sectionwise failed: {e}")
            cheating_detection = f"[CheatingDetector ERROR] {e}"

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

    def review_paper(self, pdf_path: str, query: str) -> Dict[str, Any]:
        """
        Synchronous method to review a paper from a file path.
        
        Args:
            pdf_path: Path to the PDF file
            query: Review query/question for the paper
            
        Returns:
            Dictionary containing:
            - final_summary: str - Complete evaluation in full text
            - sections: List[Dict] - Paper sections with structure
            - normalized_paper: str - Restructured full paper
            - paper_memory: str - Key points summary
            - cheating_detection: dict/str - Cheating analysis by section
            - motivation_evaluation: str - Motivation analysis
            - raw_pdf_text: str - Original extracted text
            - artifacts: Dict - Additional processing artifacts
        """
        logger.info(f"Starting paper review from file: {pdf_path}")
        
        # Read PDF file and convert to base64
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            pdf_content = base64.b64encode(pdf_bytes).decode('utf-8')
            logger.info(f"Successfully loaded PDF file: {pdf_path} ({len(pdf_bytes)} bytes)")
        except Exception as e:
            logger.error(f"Failed to read PDF file: {e}")
            raise
        
        # Run the async pipeline synchronously
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._run_sync_internal(pdf_content, query))
        
        logger.info("Paper review completed successfully.")
        return result
