"""
Unified Bias Index (UBI) Framework
Bias detection across race, profession, religion, and political ideology
"""

__version__ = "1.0.0"
__author__ = "Bias Detection Research Team"

from .data_loader import DataLoader
from .llm_connector import LLMConnector
from .baseline_manager import BaselineManager
from .pipeline import BiasDetectionPipeline

__all__ = [
    "DataLoader",
    "LLMConnector", 
    "BaselineManager",
    "BiasDetectionPipeline"
]