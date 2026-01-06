import re
from dataclasses import dataclass
import arxiv
from ..base_agent import BaseAgent
import asyncio
from ..logger import get_logger
from .paper_retrieval import PaperRetrieval, RetrievedPaper

logger = get_logger(__name__)

@dataclass
class ResemblePaper:
    title: str
    authors: str
    abstract: str
    resemblance: str
    
    @property
    def prompt(self) -> str:
        return f"<paper>\nTitle: {self.title}\nAuthors: {self.authors}\nAbstract: {self.abstract}\nResemblance Analysis: {self.resemblance}</paper>"
        
    
def extract_tag(text: str, tag: str) -> str:
    """Extract content between specified tags in the text"""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

class MotivationEvaluator(BaseAgent):
    """Agent for evaluating the motivation in research papers"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    REPORT_PROMPT = """[System Role]
You are a Senior Area Chair focusing on "Novelty and Significance" evaluation.
Your goal is to generate a comprehensive **Innovation Assessment Report**.
You must weigh the "Claims of the Manuscript" against the "Evidence of Similarity" provided in the inputs.

[Input]
1. **The Manuscript**: The full text/summary of the paper under review.
2. **Similar Works Analysis**: A list of potential overlaps found by a previous analysis step (specifically highlighting resemblances).

[Critical Constraints]
1. **Fact-Based**: Do not hallucinate. If the manuscript discusses a related work, quote the specific section/line where they differentiate themselves.
2. **Synthesized Judgement**: Do not just repeat the Similar Works Analysis. You must determine if the Manuscript *successfully defends* itself against these similarities.
3. **Tone**: Professional, objective, and critical.

[Analysis Logic]
1. **Extract Motivation**: What "Gap" do the authors claim to fill? (Look at Introduction/Related Work).
2. **Cross-Examination**:
   - Take the "Resemblances" provided in the input.
   - Check: Does the Manuscript cite these works?
   - Check: Does the Manuscript offer a *valid* technical distinction, or is it a "Strawman" argument (attacking a weak version of the problem)?
3. **Verdict**: Is the novelty "Incremental" (engineering tweak), "Application" (old method, new data), or "Substantive" (new math/theory)?

[Output Template]
Please structure the report strictly as follows:

# Innovation & Motivation Report

## 1. The Authors' Claimed Contribution
- **Core Problem**: [What are they trying to solve?]
- **Claimed Gap**: [What limitation in prior work do they claim to address? Quote from the Intro/Abstract]
- **Proposed Solution**: [Brief summary of their method]

## 2. Comparative Scrutiny (The "Trial")
*Here, we analyze how the paper stands against the identified similar works.*

**(Repeat this block for the most critical similar works provided in input)**
### vs. [Title of Similar Work]
- **Identified Overlap**: [Summarize the resemblance found in the input]
- **Manuscript's Defense**: [Did the authors cite this? How do they differentiate? e.g., "In Section 2, authors claim X is slow, whereas their method is fast."]
- **Reviewer's Assessment**: [Is this difference significant? Or is it just a minor variation?]

## 3. Novelty Verdict
- **Innovation Type**: [Incremental / Application-Oriented / Substantive]
- **Assessment**:
  [Provide a final summary. Does the paper survive the comparison? Does the existence of the similar works weaken the motivation significantly?]
  - **Strength**: ...
  - **Weakness**: ...

## 4. Key Evidence Anchors
- [List crucial sections, equations, or page numbers from the Manuscript that support your verdict]
"""

    EXTRACT_KEYWORDS_PROMPT = """[System Role] You are an expert at extracting keywords from research paper summaries.
Your task is to identify and extract the most relevant keywords that represent the main topics and themes of the paper.
The keywords are used to search for similar works in academic databases, such as arXiv.

[Critical Constraints]
1) Efficiency: Extract a concise list of 3-5 keywords that best capture the essence of the paper.
2) Avoid Redundancy: Ensure that the overlapping between keywords is minimized. The concept of one keyword should not cover another.
3) Use domain-specific terms: Include technical terms and jargon that are commonly used in the relevant research field.

[Input]
A short but concise summary of the research paper.

[Output]
A list of keywords that best represent the main topics and themes of the paper.
1) Output the keywords at the end of the response bewteen <keywords> and </keywords> tags.
2) Format the keywords as a comma-separated string, e.g., <keywords>keyword1, keyword2, keyword3</keywords>.
"""

    FILTER_RESEMBLANCE_PROMPT = """[System Role]
You are a specialist in "Genealogical Analysis of Scientific Literature".
Your task is NOT to evaluate novelty, but to **explicitly identify and articulate ALL underlying connections** between the Target Paper (Paper A) and the Reference Paper (Paper B).

[Operating Assumption]
Even if Paper A proposes improvements, assume it operates within the paradigm established or influenced by Paper B.
You must answer: "How does Paper A reflect the shadow of Paper B?"

[Input Data]
- Target Paper (A) Summary
- Reference Paper (B) Abstract

[Output Format]
First, perform a mapping analysis inside <analysis> tags.
Then, output the final structured comparison inside <resemblance> tags.

<analysis>
[Component Mapping]
- **Problem Space**: How is A's problem a specific case or variant of B's problem?
- **Method Architecture**: Map specific modules in A to their counterparts in B (e.g., "A's [Component X] acts similarly to B's [Component Y]").
- **Logic Flow**: Identify shared theoretical assumptions.
</analysis>

<resemblance>
**Core Association**: [One sentence summarizing the strongest link between A and B]

**Detailed Correspondences**:
1. **Problem Alignment**: [Explain how A's goal overlaps with B's scope]
2. **Methodological Echo**: [Describe how A's technique mirrors B's approach, using specific terms from the input]
3. **Theoretical Foundation**: [Explain the shared underlying principles]
</resemblance>
"""
    
    async def run(self, paper_summary: str) -> str:
        """
        Detect cheating in the provided research paper PDF content.

        Args:
            pdf_content: Paper content in well-formed structured text (can be full text or paper_memory)

        Returns:
            Plain text motivation evaluation string
        """
        keywords = await self._extract_keywords(paper_summary)
        
        # Use PaperRetrieval for searching
        retriever = PaperRetrieval()
        relevant_works = await retriever.search(query=" ".join(keywords), max_results=self.config.get("retrieval.max_results", 15))
        
        similar_works = await self._most_similar_works(paper_summary, relevant_works)
        motivation_report = await self._generate_report(paper_summary, similar_works)
        
        return motivation_report
    
    async def _extract_keywords(self, paper_summary: str) -> list[str]:
        """
        Extract keywords from the paper summary using the LLM.

        Args:
            paper_summary: Summary of the paper
        Returns:
            Extracted keywords as a list of strings
        """
        logger.info(f"Extracting keywords from paper summary...")

        temp = self.config.get("agents.motivation_evaluator.temperature", None)
        response = await self._call_llm_with_retry(
            model=self.model,
            messages=[
                {"role": "system", "content": self.EXTRACT_KEYWORDS_PROMPT},
                {"role": "user", "content": paper_summary}
            ],
            temperature=temp
        )

        try:
            response_text = self._get_text_from_response(response)
        except Exception as e:
            logger.error(f"Failed to parse keyword extraction response: {e}")
            raise
        keywords_text = extract_tag(response_text, "keywords")
        assert keywords_text, f"Failed to extract keywords from LLM response: {response}"
        keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
        
        logger.info(f"Extracted keywords: {keywords}")
        return keywords
    
    async def _most_similar_works(self, paper_summary: str, results: list[RetrievedPaper]) -> list[ResemblePaper]:
        """
        Find similar works based on keywords using retrieved papers.

        Args:
            paper_summary: Summary of the paper
            results: List of retrieved papers
        Returns:
            A list of similar works as ResemblePaper objects
        """
        resembling_papers = []
        semaphore = asyncio.Semaphore(self.config.get("concurrency.motivation_evaluator", 5))  # Limit concurrency

        # Create tasks for parallel processing
        async def process_result(result):
            async with semaphore:
                logger.info(f"Judging Relevance: {result.title}")
                temp = self.config.get("agents.motivation_evaluator.temperature", None)
                response = await self._call_llm_with_retry(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.FILTER_RESEMBLANCE_PROMPT},
                        {"role": "user", "content": f"Target Paper:\n{paper_summary}\n\nSimilar Work:\nTitle: {result.title}\nAuthors: {', '.join([author.name for author in result.authors])}\nAbstract: {result.summary}"}
                    ],
                    temperature=temp
                )
                try:
                    response_text = self._get_text_from_response(response)
                except Exception as e:
                    logger.error(f"Failed to parse resemblance response for '{result.title}': {e}")
                    return None

                resemblance = extract_tag(response_text, "resemblance")
                if resemblance:
                    resembling_paper = ResemblePaper(
                        title=result.title,
                        authors=", ".join([author.name for author in result.authors]),
                        abstract=result.summary,
                        resemblance=resemblance
                    )
                    logger.info(f"Finished Resemblance Analysis: {resembling_paper.title}")
                    # logger.debug(f"Resemblance Analysis: {resemblance}")
                    return resembling_paper
        
        # Process all results in parallel
        tasks = [process_result(result) for result in results]
        processed_results = await asyncio.gather(*tasks)
        
        # Filter out None values
        resembling_papers = [paper for paper in processed_results if paper is not None]
        
        return resembling_papers
    

    
    async def _generate_report(self, paper_summary: str, similar_works: list[ResemblePaper]) -> str:
        """
        Generate the motivation evaluation report using the LLM.

        Args:
            paper_summary: Summary of the paper
            similar_works: List of similar works
        Returns:
            The motivation evaluation report as a string
        """
        logger.info(f"Generating motivation evaluation report...")
        
        if not similar_works:
            logger.info("No similar works found, skipping report generation.")
            return "No evidence found that weakens the motivation of the paper."
        
        # Format similar works for the prompt
        similar_works_text = "\n\n".join([
            work.prompt
            for work in similar_works
        ])
        prompt = f"""Based on the following paper summary and similar works, evaluate the motivation of the paper:

        Paper Summary:
        {paper_summary}

        Similar Works:
        {similar_works_text}
        """

        report = await self._call_llm_with_retry(
            model=self.model,
            messages=[
                {"role": "system", "content": self.REPORT_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.get("agents.motivation_evaluator.temperature", None)
        )

        logger.info("Finished generating motivation evaluation report.")
        try:
            return self._get_text_from_response(report)
        except Exception as e:
            logger.error(f"Failed to parse motivation report: {e}")
            return f"[MOTIVATION_ERROR] {e}"
 