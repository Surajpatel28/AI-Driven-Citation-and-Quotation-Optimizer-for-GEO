"""
GEO Citation Optimizer Core Module

This module provides object-oriented components for the AI-Driven Citation
and Quotation Optimizer for Generative Engine Optimization (GEO).
"""

from .content_processor import ContentProcessor
from .geo_optimizer import GEOOptimizer
from .evaluation_engine import EvaluationEngine
from .benchmark_runner import BenchmarkRunner
from .streamlit_app import StreamlitApp
from .config import Config, get_config

__all__ = [
    # OOP classes
    'ContentProcessor',
    'GEOOptimizer', 
    'EvaluationEngine',
    'BenchmarkRunner',
    'StreamlitApp',
    'Config',
    'get_config'
]

__version__ = "2.0.0"
__author__ = "Suraj Patel"
__email__ = "suraj123patel123@gmail.com"