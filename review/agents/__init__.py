"""
Agent module for Science Arena Challenge
"""
from .base_agent import BaseAgent
from .baseline import BaseLineAgent
from .neurodong import NeuroDongAgent
from .s1 import S1Agent

__all__ = [
    'BaseAgent',
    'BaseLineAgent',
    'NeuroDongAgent',
    'S1Agent'
]

PaperReviewAgent = S1Agent