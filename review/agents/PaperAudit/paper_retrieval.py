import os
import asyncio
import arxiv
import httpx
from typing import List, Optional, Protocol, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from ..config import config

try:
    from ..logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

logger = get_logger(__name__)

@dataclass
class Author:
    name: str

@dataclass
class RetrievedPaper:
    title: str
    authors: List[Author]
    summary: str
    url: str = ""
    
    # Compatibility properties if needed
    @property
    def abstract(self) -> str:
        return self.summary

class PaperSource(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int) -> List[RetrievedPaper]:
        pass

class ArxivSource(PaperSource):
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def search(self, query: str, max_results: int) -> List[RetrievedPaper]:
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Searching arXiv (attempt {attempt + 1}/{self.max_retries})...")
                
                # Run synchronous arxiv client in a thread executor since it's blocking
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None, 
                    lambda: list(arxiv.Client().results(
                        arxiv.Search(
                            query=query,
                            max_results=max_results,
                            sort_by=arxiv.SortCriterion.Relevance
                        )
                    ))
                )
                
                retrieved_papers = []
                for result in results:
                    authors = [Author(name=a.name) for a in result.authors]
                    retrieved_papers.append(RetrievedPaper(
                        title=result.title,
                        authors=authors,
                        summary=result.summary,
                        url=result.entry_id
                    ))
                
                logger.info(f"arXiv search successful: Found {len(retrieved_papers)} papers")
                return retrieved_papers
            
            except Exception as e:
                logger.warning(f"arXiv search attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    delay = (config.get("retrieval.retry_delay_base", 2) ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All arXiv retry attempts failed")
                    # Don't raise here, let the manager handle fallback by returning empty list or raising specific error
                    # But to trigger fallback, we might want to raise or return empty. 
                    # If we return empty, it might just mean no results. 
                    # If we raise, it means service failure.
                    raise e
        return []

class SemanticScholarSource(PaperSource):
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        self.api_key = api_key
        self.max_retries = max_retries

    async def search(self, query: str, max_results: int) -> List[RetrievedPaper]:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
            
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,url"
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Searching Semantic Scholar (attempt {attempt + 1}/{self.max_retries})...")
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        self.BASE_URL, 
                        params=params, 
                        headers=headers, 
                        timeout=10.0
                    )
                    
                    if response.status_code == 403 and headers.get("x-api-key"):
                        logger.warning("Semantic Scholar API key invalid or forbidden. Retrying without key...")
                        headers.pop("x-api-key", None)
                        # Don't count this as a retry attempt, try immediately
                        continue

                    if response.status_code == 429:
                        logger.warning("Rate limit exceeded for Semantic Scholar")
                        raise Exception("Rate limit exceeded")
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    papers = data.get("data", [])
                    retrieved_papers = []
                    
                    for paper in papers:
                        if not paper.get("title") or not paper.get("abstract"):
                            continue
                            
                        authors = [Author(name=a["name"]) for a in paper.get("authors", [])]
                        retrieved_papers.append(RetrievedPaper(
                            title=paper["title"],
                            authors=authors,
                            summary=paper["abstract"],
                            url=paper.get("url", "")
                        ))
                        
                    logger.info(f"Semantic Scholar search successful: Found {len(retrieved_papers)} papers")
                    return retrieved_papers

            except Exception as e:
                logger.warning(f"Semantic Scholar search attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    delay = (config.get("retrieval.retry_delay_base", 2) ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All Semantic Scholar retry attempts failed")
                    raise e
        return []

class PaperRetrieval:
    def __init__(self):
        self.sources: List[PaperSource] = []
        
        # Initialize sources
        # Priority 2: arXiv
        arxiv_retries = config.get("retrieval.arxiv_max_retries", 3)
        self.sources.append(ArxivSource(max_retries=arxiv_retries))

        # Priority 1: Semantic Scholar
        ss_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        # Public API is more likely to be rate limited, so we give it more retries
        ss_retries = config.get("retrieval.semantic_scholar_max_retries", 5)
        self.sources.append(SemanticScholarSource(api_key=ss_api_key, max_retries=ss_retries))
        
        
    async def search(self, query: str, max_results: int) -> List[RetrievedPaper]:
        """
        Search for papers using registered sources with fallback.
        """
        errors = []
        
        for source in self.sources:
            source_name = source.__class__.__name__
            try:
                results = await source.search(query, max_results)
                if results:
                    return results
                else:
                    logger.info(f"{source_name} returned no results, trying next source...")
            except Exception as e:
                logger.warning(f"{source_name} failed: {e}, trying next source...")
                errors.append(f"{source_name}: {str(e)}")
                continue
                
        logger.error(f"All paper retrieval sources failed. Errors: {errors}")
        return []

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    async def main():
        print("Testing PaperRetrieval...")
        retriever = PaperRetrieval()
        
        # Test query
        query = "Large Language Models for Software Engineering"
        print(f"Searching for: {query}")
        
        try:
            results = await retriever.search(query, max_results=15)
            
            print(f"\nFound {len(results)} papers:")
            for i, paper in enumerate(results, 1):
                print(f"{i}. {paper.title}")
                print(f"   Authors: {', '.join(a.name for a in paper.authors)}")
                print(f"   URL: {paper.url}")
                print("-" * 50)
        except Exception as e:
            print(f"Search failed: {e}")

    asyncio.run(main())