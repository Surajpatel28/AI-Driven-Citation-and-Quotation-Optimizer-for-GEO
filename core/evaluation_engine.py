"""
Evaluation Engine Module

This module handles the automated evaluation of optimized content,
providing comprehensive metrics and analysis for GEO optimization quality.
"""

import re
import os
import pandas as pd
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Data class to hold evaluation metrics."""
    has_citation: bool
    has_statistic: bool
    has_quote: bool
    has_structure: bool
    word_count: int
    

class EvaluationEngine:
    """
    Comprehensive evaluation engine for GEO optimization outputs.
    
    This class provides automated evaluation of optimized content to assess
    the presence and quality of citations, statistics, quotes, and structure.
    """
    
    def __init__(self):
        """Initialize the EvaluationEngine."""
        self.logger = self._setup_logging()
        self._compile_evaluation_patterns()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _compile_evaluation_patterns(self) -> None:
        """Compile regex patterns for evaluation."""
        # Citation patterns
        self.citation_pattern = re.compile(
            r'(https?://|Citation:|Source:|\[\d+\])', 
            re.IGNORECASE
        )
        
        # Statistics patterns
        self.statistic_pattern = re.compile(
            r'\b\d+[,.]?\d*\s?(%|percent|million|billion|thousand|[A-Za-z]+)?\b'
        )
        
        # Quote patterns
        self.quote_patterns = [
            re.compile(r'"[^"]+"|"[^"]+"'),  # Quoted text
            re.compile(r'\bsaid\b', re.IGNORECASE),  # Attribution
            re.compile(r'\baccording to\b', re.IGNORECASE)  # According to
        ]
        
        # Structure patterns
        self.structure_patterns = [
            re.compile(r'^\s*#{1,6}\s', re.MULTILINE),  # Markdown headings
            re.compile(r'^\s*[\-\*\•]\s', re.MULTILINE),  # Bullet points
            re.compile(r'^\s*\d+\.\s', re.MULTILINE),  # Numbered lists
            re.compile(r'\n\s*\n'),  # Paragraph breaks
            re.compile(
                r'\b(introduction|conclusion|summary|background|results|discussion|references)\b', 
                re.IGNORECASE
            )  # Section titles
        ]
    
    def has_citation(self, text: str) -> bool:
        """
        Check if text contains citations.
        
        Args:
            text: Text to evaluate
            
        Returns:
            True if citations are present
        """
        return bool(self.citation_pattern.search(text))
    
    def has_statistic(self, text: str) -> bool:
        """
        Check if text contains statistics.
        
        Args:
            text: Text to evaluate
            
        Returns:
            True if statistics are present
        """
        return (bool(self.statistic_pattern.search(text)) or 
                'according to' in text.lower())
    
    def has_quote(self, text: str) -> bool:
        """
        Check if text contains quotes or attributions.
        
        Args:
            text: Text to evaluate
            
        Returns:
            True if quotes are present
        """
        return any(pattern.search(text) for pattern in self.quote_patterns)
    
    def has_structure(self, text: str) -> bool:
        """
        Check if text has proper structure.
        
        Args:
            text: Text to evaluate
            
        Returns:
            True if structure is present
        """
        return any(pattern.search(text) for pattern in self.structure_patterns)
    
    def count_words(self, text: str) -> int:
        """
        Count words in text.
        
        Args:
            text: Text to count words in
            
        Returns:
            Word count
        """
        return len(str(text).split())
    
    def evaluate_single_output(self, text: str) -> EvaluationMetrics:
        """
        Evaluate a single output text.
        
        Args:
            text: Text to evaluate
            
        Returns:
            EvaluationMetrics object with results
        """
        return EvaluationMetrics(
            has_citation=self.has_citation(text),
            has_statistic=self.has_statistic(text),
            has_quote=self.has_quote(text),
            has_structure=self.has_structure(text),
            word_count=self.count_words(text)
        )
    
    def evaluate_batch(self, texts: List[str]) -> List[EvaluationMetrics]:
        """
        Evaluate a batch of output texts.
        
        Args:
            texts: List of texts to evaluate
            
        Returns:
            List of EvaluationMetrics objects
        """
        results = []
        for text in texts:
            try:
                results.append(self.evaluate_single_output(text))
            except Exception as e:
                self.logger.error(f"Error evaluating text: {e}")
                # Return default metrics for failed evaluation
                results.append(EvaluationMetrics(
                    has_citation=False,
                    has_statistic=False,
                    has_quote=False,
                    has_structure=False,
                    word_count=0
                ))
        return results
    
    def evaluate_from_dataframe(self, df: pd.DataFrame, 
                               output_column: str = 'optimized_output') -> pd.DataFrame:
        """
        Evaluate outputs from a pandas DataFrame.
        
        Args:
            df: DataFrame containing outputs
            output_column: Column name containing the outputs to evaluate
            
        Returns:
            DataFrame with evaluation results
        """
        if output_column not in df.columns:
            raise ValueError(f"Column '{output_column}' not found in DataFrame.")
        
        # Extract texts and evaluate
        texts = df[output_column].astype(str).tolist()
        metrics = self.evaluate_batch(texts)
        
        # Convert to DataFrame
        results_data = []
        for metric in metrics:
            results_data.append({
                'has_citation': metric.has_citation,
                'has_statistic': metric.has_statistic,
                'has_quote': metric.has_quote,
                'structure_check': metric.has_structure,
                'word_count': metric.word_count
            })
        
        return pd.DataFrame(results_data)
    
    def calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate summary statistics from evaluation results.
        
        Args:
            df: DataFrame with evaluation results
            
        Returns:
            Dictionary with summary statistics
        """
        return {
            'citation_presence_pct': df['has_citation'].mean() * 100,
            'statistic_presence_pct': df['has_statistic'].mean() * 100,
            'quote_presence_pct': df['has_quote'].mean() * 100,
            'structure_presence_pct': df['structure_check'].mean() * 100,
            'average_word_count': df['word_count'].mean(),
            'total_outputs': len(df)
        }
    
    def generate_evaluation_report(self, df: pd.DataFrame) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            df: DataFrame with evaluation results
            
        Returns:
            Formatted evaluation report
        """
        stats = self.calculate_summary_statistics(df)
        
        report = f"""
        GEO Optimization Evaluation Report
        ================================
        
        Total Outputs Evaluated: {stats['total_outputs']}
        
        Content Quality Metrics:
        ----------------------
        Citation Presence: {stats['citation_presence_pct']:.1f}%
        Statistic Presence: {stats['statistic_presence_pct']:.1f}%
        Quote Presence: {stats['quote_presence_pct']:.1f}%
        Structure Presence: {stats['structure_presence_pct']:.1f}%
        
        Content Length:
        --------------
        Average Word Count: {stats['average_word_count']:.1f}
        
        Overall Score: {(stats['citation_presence_pct'] + stats['statistic_presence_pct'] + stats['quote_presence_pct'] + stats['structure_presence_pct']) / 4:.1f}%
        """
        
        return report
    
    def evaluate_outputs_from_file(self, input_path: str, 
                                  output_path: str,
                                  output_column: str = 'optimized_output') -> None:
        """
        Evaluate outputs from a CSV file and save results.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save evaluation results
            output_column: Column name containing outputs to evaluate
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        try:
            # Load data
            df = pd.read_csv(input_path)
            self.logger.info(f"Loaded {len(df)} outputs from {input_path}")
            
            # Evaluate
            results_df = self.evaluate_from_dataframe(df, output_column)
            
            # Calculate and display summary statistics
            stats = self.calculate_summary_statistics(results_df)
            
            self.logger.info(f"Citation presence: {stats['citation_presence_pct']:.1f}%")
            self.logger.info(f"Statistic presence: {stats['statistic_presence_pct']:.1f}%")
            self.logger.info(f"Quote presence: {stats['quote_presence_pct']:.1f}%")
            self.logger.info(f"Structure presence: {stats['structure_presence_pct']:.1f}%")
            self.logger.info(f"Average word count: {stats['average_word_count']:.1f}")
            
            # Save results
            results_df.to_csv(output_path, index=False)
            self.logger.info(f"Evaluation results saved to {output_path}")
            
            # Generate and save report
            report_path = output_path.replace('.csv', '_report.txt')
            with open(report_path, 'w') as f:
                f.write(self.generate_evaluation_report(results_df))
            self.logger.info(f"Evaluation report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise


# Legacy function for backward compatibility
def evaluate_outputs(input_path: str, output_path: str) -> None:
    """
    Legacy function wrapper for backward compatibility.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save evaluation results
    """
    evaluator = EvaluationEngine()
    evaluator.evaluate_outputs_from_file(input_path, output_path)
