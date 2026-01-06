# agents/PaperAudit/audit_agent.py
import json
import asyncio
import re
from typing import Any, AsyncGenerator, Dict, List, Optional
from collections import OrderedDict
from pathlib import Path

from ..base_agent import BaseAgent
from ..logger import get_logger, get_artifact_logger
from .cheating_detector import CheatingDetector
from .motivation_evaluator import MotivationEvaluator
from .summarizer import Summarizer
from .paper_memory_summarizer import PaperMemorySummarizer

logger = get_logger(__name__)
artifact_logger = get_artifact_logger("audit_artifacts")


class AuditAgent(BaseAgent):
    def __init__(
        self,
        *,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        shared_kwargs = dict(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        self.cheating_detector = CheatingDetector(**shared_kwargs)
        self.motivation_evaluator = MotivationEvaluator(**shared_kwargs)
        self.summarizer = Summarizer(**shared_kwargs)
        self.paper_memory_summarizer = PaperMemorySummarizer(**shared_kwargs)

    def _parse_sse_chunk(self, sse_string: str) -> str:
        try:
            if sse_string.startswith("data: "):
                sse_string = sse_string[6:]
            sse_string = sse_string.strip()

            if sse_string == "[DONE]":
                return ""

            data = json.loads(sse_string)
            if (
                isinstance(data, dict)
                and data.get("choices")
                and len(data["choices"]) > 0
                and data["choices"][0].get("delta")
                and data["choices"][0]["delta"].get("content") is not None
            ):
                return data["choices"][0]["delta"]["content"]

            return ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.debug(f"Failed to parse SSE chunk: {e}")
            return ""

    def _create_chunk(self, content: str) -> str:
        response_data = {"object": "chat.completion.chunk", "choices": [{"delta": {"content": content}}]}
        return f"data: {json.dumps(response_data)}\n\n"

    def _build_sections_from_blocks(self, blocks: List[Dict[str, Any]], enable_mm: bool) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []
        current_title: Optional[str] = None
        buffer: List[Dict[str, Any]] = []

        def flush() -> None:
            nonlocal buffer, current_title
            if not buffer:
                return
            content = self.blocks_to_prompt_content(buffer, enable_mm=enable_mm)
            content_text = self.blocks_to_text(buffer, enable_mm=False)
            if (content_text if isinstance(content, list) else content).strip():
                sections.append(
                    {
                        "title": current_title or "Unknown Section",
                        "content": content,
                        "content_text": content_text,
                    }
                )
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
            sections.append(
                {
                    "title": "Full Paper",
                    "content": self.blocks_to_prompt_content(blocks, enable_mm=enable_mm),
                    "content_text": self.blocks_to_text(blocks, enable_mm=False),
                }
            )

        merged: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        for sec in sections:
            title = (sec.get("title") or "Unknown Section").strip() or "Unknown Section"
            if title not in merged:
                merged[title] = {
                    "title": title,
                    "content": sec["content"],
                    "content_text": sec.get("content_text", ""),
                }
            else:
                if isinstance(merged[title]["content"], list):
                    merged[title]["content"].extend(sec["content"])
                else:
                    merged[title]["content"] = merged[title]["content"].rstrip() + "\n\n" + str(sec["content"]).lstrip()
                merged[title]["content_text"] = merged[title]["content_text"].rstrip() + "\n\n" + sec.get(
                    "content_text", ""
                ).lstrip()

        return list(merged.values())

    @staticmethod
    def _combine_sections_content(sections: List[Dict[str, Any]], enable_mm: bool) -> Any:
        if not enable_mm:
            return "\n\n".join(
                f"# {sec['title']}\n{sec.get('content_text') or sec.get('content', '')}" for sec in sections
            )

        parts: List[Dict[str, Any]] = []
        for sec in sections:
            parts.append({"type": "text", "text": f"# {sec['title']}"})
            content = sec.get("content")
            if isinstance(content, list):
                parts.extend(content)
            elif isinstance(content, str):
                parts.append({"type": "text", "text": content})
        return parts

    async def _run_with_progress(self, task_coro, description: str, result_container: List[Any]):
        task = asyncio.create_task(task_coro)
        yield self._create_chunk(f"> {description}...\n> ")

        while not task.done():
            done, _ = await asyncio.wait([task], timeout=3.0)
            if not done:
                yield self._create_chunk(".")

        yield self._create_chunk("\n")
        result_container.append(task.result())

    def _write_cache_text(self, cache_dir: Optional[Path], filename: str, content: str) -> None:
        """
        Stage-by-stage persistence:
        - write only if cache_dir is provided
        - write only if content is non-empty
        - do NOT overwrite existing files (prevents accidental clobber)
        """
        if cache_dir is None:
            return
        if content is None:
            return
        content = str(content)
        if not content.strip():
            return
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            p = cache_dir / filename
            if p.exists():
                return
            p.write_text(content, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write cache file {filename}: {e}")

    async def run(
        self,
        paper_json: Any,
        query: str,
        *,
        enable_mm: bool = False,
        enable_cheating_detection: bool = True,
        enable_motivation: bool = True,
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError("Use review_paper/_run_sync_internal for batch runner; keep SSE path separate if needed.")

    async def _run_sync_internal(
        self,
        paper_json: Any,
        query: str,
        *,
        enable_mm: bool = False,
        enable_cheating_detection: bool = True,
        enable_motivation: bool = True,
        reuse_cache: bool = False,
        cache_dir: Optional[Path] = None,
        model_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        # ---------------- cache readers ----------------
        def _read_text(p: Path) -> Optional[str]:
            try:
                if p.exists():
                    return p.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed reading cache file: {p} :: {e}")
            return None

        cache_path = Path(cache_dir) if cache_dir is not None else None

        # Only read caches that we might use
        cached_baseline = _read_text(cache_path / "baseline_review.txt") if (reuse_cache and cache_path) else None

        # final review is ALWAYS recomputed, so do NOT read it from cache
        cached_final = None

        cached_memory = _read_text(cache_path / "paper_memory.txt") if (reuse_cache and cache_path) else None
        cached_cheat = _read_text(cache_path / "cheat_report.txt") if (reuse_cache and cache_path) else None
        cached_motiv = _read_text(cache_path / "motivation_report.txt") if (reuse_cache and cache_path) else None

        # ---------------- artifacts ----------------
        artifacts: Dict[str, Any] = {}

        def _save_text(name: str, text: str):
            artifact_logger.info(f"\n=== {name} ===\n{text}")
            artifacts[name] = text

        def _save_json(name: str, obj: Any):
            artifact_logger.info(f"\n=== {name} ===\n{json.dumps(obj, ensure_ascii=False, indent=2)}")
            artifacts[name] = obj

        # ---------------- Step 0: normalize blocks & sections ----------------
        blocks = self.prepare_paper_blocks(paper_json)
        logger.info(f"Normalized paper blocks: {len(blocks)} items")

        plain_blocks_text = self.blocks_to_text(blocks, enable_mm=False)
        _save_text("plain_blocks_text", plain_blocks_text)
        _save_json("paper_blocks", blocks)

        logger.info("Aggregating sections from paper JSON (no LLM structuring)...")
        sections: List[Dict[str, Any]] = self._build_sections_from_blocks(blocks, enable_mm=enable_mm)
        for i, sec in enumerate(sections, start=1):
            title = sec.get("title", "")
            content_len = len(sec.get("content_text", "")) if isinstance(sec.get("content"), list) else len(
                str(sec.get("content", ""))
            )
            logger.info(f"  - Section {i}: {title} (len={content_len} chars)")
        _save_json("sections", sections)

        normalized_paper_content = self._combine_sections_content(sections, enable_mm=enable_mm)
        if enable_mm:
            logger.info("Normalized paper content assembled as multimodal parts")
        else:
            logger.info(f"Normalized paper text length: {len(normalized_paper_content)} characters")
        _save_text("normalized_paper", self.blocks_to_text(blocks, enable_mm=False))

        # ---------------- Step 1: Baseline review (NO helper reports) ----------------
        baseline_review: str = ""
        if reuse_cache and cached_baseline:
            baseline_review = cached_baseline
            logger.info(f"Reusing cached baseline review (len={len(baseline_review)})")
        else:
            logger.info("Starting BASELINE review (no helper reports)...")
            baseline_chunks: List[str] = []
            try:
                async for chunk in self.summarizer.run_baseline(normalized_paper_content, query):
                    content = self._parse_sse_chunk(chunk)
                    baseline_chunks.append(content)
            except Exception as e:
                logger.error(f"Summarizer.run_baseline failed: {e}")
                baseline_chunks = [f"[BASELINE_REVIEW_ERROR] {e}"]
            baseline_review = "".join(baseline_chunks)

        _save_text("baseline_review", baseline_review)
        self._write_cache_text(cache_dir, "baseline_review.txt", baseline_review)

        # final review is always recomputed; no cache hit logic needed
        final_review: str = ""

        # ---------------- Step 2: paper memory (ONLY if motivation enabled) ----------------
        paper_memory: str = ""
        need_memory = True

        if not need_memory:
            logger.info("Skipping paper memory summarization (motivation disabled).")
        else:
            if reuse_cache and cached_memory:
                paper_memory = cached_memory
                logger.info(f"Reusing cached paper_memory (len={len(paper_memory)})")
            else:
                logger.info("Starting paper memory summarization (no cache hit)...")
                try:
                    paper_memory_chunks: List[str] = []
                    async for chunk in self.paper_memory_summarizer.run(
                        normalized_paper_content, section_titles=[s.get("title", "") for s in sections]
                    ):
                        paper_memory_chunks.append(chunk)
                    paper_memory = "".join(paper_memory_chunks)
                    logger.info(f"Paper memory summarization completed, length={len(paper_memory)}")
                except TypeError:
                    try:
                        paper_memory = await self.paper_memory_summarizer.run(
                            normalized_paper_content, section_titles=[s.get("title", "") for s in sections]
                        )
                        logger.info(f"Paper memory summarization completed, length={len(paper_memory)}")
                    except Exception as e:
                        logger.error(f"Paper memory summarization failed: {e}")
                        paper_memory = f"[MEMORY_ERROR] {e}"
                except Exception as e:
                    logger.error(f"Paper memory summarization failed: {e}")
                    paper_memory = f"[MEMORY_ERROR] {e}"

            _save_text("paper_memory", paper_memory)
            self._write_cache_text(cache_dir, "paper_memory.txt", paper_memory)

        # ---------------- Step 3: cheating + motivation (reusable; DO NOT write placeholders) ----------------
        cheating_detection: Any = None
        motivation_evaluation: Any = None

        # cheating (global_review)
        if not enable_cheating_detection:
            logger.info("Skipping cheating detection (disabled).")
        else:
            if reuse_cache and cached_cheat:
                cheating_detection = cached_cheat
                logger.info(f"Reusing cached cheating report (len={len(str(cheating_detection))})")
            else:
                logger.info("Starting global cheating review via CheatingDetector (no cache hit)...")
                try:
                    paper_blocks_payload = self.blocks_to_prompt_content(blocks, enable_mm=enable_mm)
                    if isinstance(paper_blocks_payload, str):
                        paper_blocks_payload = [{"type": "text", "text": paper_blocks_payload}]

                    cheating_detection = await self.cheating_detector.run_review(
                        "global_review",
                        paper_blocks=paper_blocks_payload,
                    )
                    logger.info("CheatingDetector global_review completed.")
                except Exception as e:
                    logger.error(f"CheatingDetector global_review failed: {e}")
                    cheating_detection = f"[CheatingDetector ERROR] {e}"

            _save_text("cheating_detection", str(cheating_detection))
            self._write_cache_text(cache_dir, "cheat_report.txt", str(cheating_detection))

        # motivation
        if not enable_motivation:
            logger.info("Skipping motivation evaluation (disabled).")
        else:
            # ensure paper_memory exists if motivation enabled
            if not paper_memory:
                if reuse_cache and cached_memory:
                    paper_memory = cached_memory
                else:
                    logger.info("Motivation enabled but paper_memory missing; recomputing paper_memory...")
                    try:
                        paper_memory_chunks: List[str] = []
                        async for chunk in self.paper_memory_summarizer.run(
                            normalized_paper_content, section_titles=[s.get("title", "") for s in sections]
                        ):
                            paper_memory_chunks.append(chunk)
                        paper_memory = "".join(paper_memory_chunks)
                    except TypeError:
                        paper_memory = await self.paper_memory_summarizer.run(
                            normalized_paper_content, section_titles=[s.get("title", "") for s in sections]
                        )
                    _save_text("paper_memory", paper_memory)
                    self._write_cache_text(cache_dir, "paper_memory.txt", paper_memory)

            if reuse_cache and cached_motiv:
                motivation_evaluation = cached_motiv
                logger.info(f"Reusing cached motivation report (len={len(str(motivation_evaluation))})")
            else:
                logger.info("Starting motivation evaluation (no cache hit)...")
                try:
                    motivation_chunks: List[str] = []
                    async for chunk in self.motivation_evaluator.run(paper_memory):
                        motivation_chunks.append(chunk)
                    motivation_evaluation = "".join(motivation_chunks)
                    logger.info(f"Motivation evaluation completed, length={len(str(motivation_evaluation))}")
                except TypeError:
                    try:
                        motivation_evaluation = await self.motivation_evaluator.run(paper_memory)
                        logger.info(f"Motivation evaluation completed, length={len(str(motivation_evaluation))}")
                    except Exception as e:
                        logger.error(f"Motivation evaluation failed: {e}")
                        motivation_evaluation = f"[MOTIVATION_ERROR] {e}"
                except Exception as e:
                    logger.error(f"Motivation evaluation failed: {e}")
                    motivation_evaluation = f"[MOTIVATION_ERROR] {e}"

            _save_text("motivation_evaluation", str(motivation_evaluation))
            self._write_cache_text(cache_dir, "motivation_report.txt", str(motivation_evaluation))

        # ---------------- Step 4: Refine review (baseline + helper reports as attention cues) ----------------
        logger.info("Starting REFINED review (update baseline using helper reports as attention cues)...")
        refined_chunks: List[str] = []
        try:
            async for chunk in self.summarizer.run_refine(
                normalized_paper_content,
                query,
                baseline_review=baseline_review,
                paper_memory=paper_memory or "",
                cheating_report=str(cheating_detection or ""),
                motivation_report=str(motivation_evaluation or ""),
            ):
                content = self._parse_sse_chunk(chunk)
                refined_chunks.append(content)
        except Exception as e:
            logger.error(f"Summarizer.run_refine failed: {e}")
            refined_chunks = [f"[REFINE_REVIEW_ERROR] {e}"]
        final_review = "".join(refined_chunks)

        _save_text("final_review", final_review)

        # write final review with tag in filename (no overwrite)
        tag = (model_tag or "").strip()
        tag_safe = re.sub(r"[^0-9A-Za-z_-]+", "_", tag) if tag else ""
        final_name = f"final_review_{tag_safe}.txt" if tag_safe else "final_review.txt"
        self._write_cache_text(cache_dir, final_name, final_review)

        logger.info("Refined review completed.")

        return {
            "final_assessment": final_review,
            "baseline_review": baseline_review,
            "sections": sections,
            "normalized_paper": normalized_paper_content,
            "paper_memory": paper_memory,  # may be "" if motivation disabled
            "cheating_detection": cheating_detection,  # may be None if disabled
            "motivation_evaluation": motivation_evaluation,  # may be None if disabled
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
        reuse_cache: bool = False,
        cache_dir: Optional[Path] = None,
        model_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self._run_sync_internal(
                paper_json,
                query,
                enable_mm=enable_mm,
                enable_cheating_detection=enable_cheating_detection,
                enable_motivation=enable_motivation,
                reuse_cache=reuse_cache,
                cache_dir=cache_dir,
                model_tag=model_tag,
            )
        )
        logger.info("Paper review completed successfully.")
        return result
