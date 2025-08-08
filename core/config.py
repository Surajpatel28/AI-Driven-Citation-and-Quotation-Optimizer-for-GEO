"""
Configuration Module for OOP Architecture

This module provides configuration settings and constants for the
object-oriented implementation of the GEO Citation Optimizer.
"""

import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.3
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"
    tokenizer_model: str = "google/flan-t5-large"


@dataclass
class ProcessingConfig:
    """Configuration for content processing."""
    max_url_content_length: int = 5000
    sentence_batch_size: int = 16
    top_sentences_count: int = 3
    min_sentence_length: int = 8


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    default_output_column: str = "optimized_output"
    feature_columns: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                'has_citation', 
                'has_statistic', 
                'has_quote', 
                'structure_check'
            ]


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    dataset_name: str = "GEO-Optim/geo-bench"
    dataset_split: str = "test"
    default_sample_size: int = 10
    output_directory: str = "result_from_benchmark"
    outputs_filename: str = "outputs.csv"
    evaluation_filename: str = "eval_report.csv"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    error_log_file: str = "error.log"
    enable_timing_logs: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    app_title: str = "AI-Driven Citation & Quotation Optimizer"
    page_layout: str = "wide"
    enable_legacy_mode: bool = True
    enable_oop_mode: bool = True
    default_architecture: str = "OOP"  # or "Legacy"


class Config:
    """Main configuration class that combines all settings."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.processing = ProcessingConfig()
        self.evaluation = EvaluationConfig()
        self.benchmark = BenchmarkConfig()
        self.logging = LoggingConfig()
        self.app = AppConfig()
        
        # Load environment overrides
        self._load_environment_overrides()
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""
        # Model config overrides
        gemini_model = os.getenv("GEMINI_MODEL")
        if gemini_model:
            self.model.gemini_model = gemini_model
        
        gemini_temp = os.getenv("GEMINI_TEMPERATURE")
        if gemini_temp:
            try:
                self.model.gemini_temperature = float(gemini_temp)
            except ValueError:
                pass
        
        # Processing config overrides
        max_url_length = os.getenv("MAX_URL_CONTENT_LENGTH")
        if max_url_length:
            try:
                self.processing.max_url_content_length = int(max_url_length)
            except ValueError:
                pass
        
        sentence_batch = os.getenv("SENTENCE_BATCH_SIZE")
        if sentence_batch:
            try:
                self.processing.sentence_batch_size = int(sentence_batch)
            except ValueError:
                pass
        
        top_sentences = os.getenv("TOP_SENTENCES_COUNT")
        if top_sentences:
            try:
                self.processing.top_sentences_count = int(top_sentences)
            except ValueError:
                pass
        
        min_sentence = os.getenv("MIN_SENTENCE_LENGTH")
        if min_sentence:
            try:
                self.processing.min_sentence_length = int(min_sentence)
            except ValueError:
                pass
        
        # Benchmark config overrides
        benchmark_sample = os.getenv("BENCHMARK_SAMPLE_SIZE")
        if benchmark_sample:
            try:
                self.benchmark.default_sample_size = int(benchmark_sample)
            except ValueError:
                pass
        
        benchmark_dataset = os.getenv("BENCHMARK_DATASET")
        if benchmark_dataset:
            self.benchmark.dataset_name = benchmark_dataset
        
        # App config overrides
        default_arch = os.getenv("DEFAULT_ARCHITECTURE")
        if default_arch and default_arch in ["OOP", "Legacy"]:
            self.app.default_architecture = default_arch
        
        enable_legacy = os.getenv("ENABLE_LEGACY_MODE")
        if enable_legacy:
            self.app.enable_legacy_mode = enable_legacy.lower() == "true"
        
        enable_oop = os.getenv("ENABLE_OOP_MODE")
        if enable_oop:
            self.app.enable_oop_mode = enable_oop.lower() == "true"
        
        page_layout = os.getenv("PAGE_LAYOUT")
        if page_layout and page_layout in ["centered", "wide"]:
            self.app.page_layout = page_layout
        
        # Logging config overrides
        log_level = os.getenv("LOG_LEVEL")
        if log_level and log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            self.logging.log_level = log_level
        
        enable_timing = os.getenv("ENABLE_TIMING_LOGS")
        if enable_timing:
            self.logging.enable_timing_logs = enable_timing.lower() == "true"
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.model
    
    def get_processing_config(self) -> ProcessingConfig:
        """Get processing configuration."""
        return self.processing
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        return self.evaluation
    
    def get_benchmark_config(self) -> BenchmarkConfig:
        """Get benchmark configuration."""
        return self.benchmark
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.logging
    
    def get_app_config(self) -> AppConfig:
        """Get application configuration."""
        return self.app
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'processing': self.processing.__dict__,
            'evaluation': self.evaluation.__dict__,
            'benchmark': self.benchmark.__dict__,
            'logging': self.logging.__dict__,
            'app': self.app.__dict__
        }


# Global configuration instance
config = Config()


# Convenience functions for accessing configuration
def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return config.get_model_config()


def get_processing_config() -> ProcessingConfig:
    """Get processing configuration."""
    return config.get_processing_config()


def get_evaluation_config() -> EvaluationConfig:
    """Get evaluation configuration."""
    return config.get_evaluation_config()


def get_benchmark_config() -> BenchmarkConfig:
    """Get benchmark configuration."""
    return config.get_benchmark_config()


def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return config.get_logging_config()


def get_app_config() -> AppConfig:
    """Get application configuration."""
    return config.get_app_config()
