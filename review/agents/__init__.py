"""
Agent module for paper review system
"""
from .base_agent import BaseAgent
from .PaperAudit import AuditAgent
from .deepreviewer import DeepReviewerAgent

__all__ = [
    "BaseAgent",
    "AuditAgent",
    "DeepReviewerAgent"
]