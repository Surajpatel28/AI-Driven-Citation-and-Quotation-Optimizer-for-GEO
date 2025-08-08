"""
Benchmark Runner Module

This module handles benchmarking operations using the GEO-Bench dataset
to evaluate the performance of the optimization system.
"""

import sys
import os
import random
import pandas as pd
import logging
from typing import List, Dict, Optional
from datasets import load_dataset

from .geo_optimizer import GEOOptimizer
from .evaluation_engine import EvaluationEngine


class BenchmarkRunner:
    """
    Benchmark runner for the GEO Citation Optimizer.
    
    This class handles loading the GEO-Bench dataset, running optimizations,
    and collecting results for performance evaluation.
    """
    
    def __init__(self, 
                 optimizer: Optional[GEOOptimizer] = None,
                 evaluator: Optional[EvaluationEngine] = None,
                 dataset_name: str = "GEO-Optim/geo-bench",
                 dataset_split: str = "test"):
        """
        Initialize the BenchmarkRunner.
        
        Args:
            optimizer: GEOOptimizer instance (creates new if None)
            evaluator: EvaluationEngine instance (creates new if None)
            dataset_name: Name of the HuggingFace dataset
            dataset_split: Dataset split to use
        """
        self.logger = self._setup_logging()
        self.optimizer = optimizer or GEOOptimizer()
        self.evaluator = evaluator or EvaluationEngine()
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.dataset = None
        self._load_dataset()
    
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
    
    def _load_dataset(self) -> None:
        """Load the GEO-Bench dataset."""
        try:
            self.logger.info(f"Loading dataset: {self.dataset_name}")
            ds = load_dataset(self.dataset_name, self.dataset_split)
            self.dataset = list(ds[self.dataset_split])
            self.logger.info(f"Loaded {len(self.dataset)} items from dataset")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            self.dataset = []
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import nltk
            import spacy
            return True
        except ImportError as e:
            self.logger.error(f"Required dependency not installed: {e}")
            self.logger.error("Please run: pip install -r requirements.txt")
            return False
    
    def sample_queries(self, n: int = 10) -> List[Dict]:
        """
        Sample random queries from the dataset.
        
        Args:
            n: Number of queries to sample
            
        Returns:
            List of sampled query dictionaries
        """
        if not self.dataset:
            self.logger.warning("Dataset not available for sampling")
            return []
        
        if n > len(self.dataset):
            self.logger.warning(f"Requested {n} samples but only {len(self.dataset)} available")
            n = len(self.dataset)
        
        return random.sample(self.dataset, n)
    
    def run_single_optimization(self, item: Dict) -> Dict:
        """
        Run optimization on a single dataset item.
        
        Args:
            item: Dataset item dictionary
            
        Returns:
            Dictionary with optimization results
        """
        query = item.get('query', '')
        tags = item.get('tags', [])
        sources = item.get('sources', [])
        
        try:
            optimized = self.optimizer.optimize_content(query)
            
            return {
                'query': query,
                'tags': tags,
                'sources': sources,
                'optimized_output': optimized,
                'optimization_status': 'success'
            }
        except Exception as e:
            self.logger.error(f"Optimization failed for query: {query[:50]}... Error: {e}")
            return {
                'query': query,
                'tags': tags,
                'sources': sources,
                'optimized_output': f"Optimization failed: {str(e)}",
                'optimization_status': 'failed'
            }
    
    def run_benchmark(self, n: int = 10, save_results: bool = True) -> pd.DataFrame:
        """
        Run benchmark on n random queries.
        
        Args:
            n: Number of queries to benchmark
            save_results: Whether to save results to CSV
            
        Returns:
            DataFrame with benchmark results
        """
        if not self._check_dependencies():
            raise RuntimeError("Required dependencies not available")
        
        if not self.dataset:
            raise RuntimeError("Dataset not loaded")
        
        self.logger.info(f"Starting benchmark with {n} queries")
        
        # Sample queries
        queries = self.sample_queries(n)
        results = []
        
        # Run optimizations
        for i, item in enumerate(queries, 1):
            self.logger.info(f"Processing query {i}/{n}")
            result = self.run_single_optimization(item)
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results if requested
        if save_results:
            output_dir = 'result_from_benchmark'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'outputs.csv')
            df.to_csv(output_path, index=False)
            self.logger.info(f"Benchmark results saved to {output_path}")
        
        return df
    
    def run_full_benchmark_with_evaluation(self, n: int = 10) -> Dict[str, str]:
        """
        Run complete benchmark including evaluation.
        
        Args:
            n: Number of queries to benchmark
            
        Returns:
            Dictionary with output and evaluation file paths
        """
        # Run benchmark
        df = self.run_benchmark(n, save_results=True)
        
        # Prepare paths
        output_dir = 'result_from_benchmark'
        output_path = os.path.join(output_dir, 'outputs.csv')
        eval_path = os.path.join(output_dir, 'eval_report.csv')
        
        # Run evaluation
        try:
            self.evaluator.evaluate_outputs_from_file(output_path, eval_path)
            self.logger.info("Benchmark and evaluation completed successfully")
            
            return {
                'outputs_path': output_path,
                'evaluation_path': eval_path,
                'status': 'success'
            }
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {
                'outputs_path': output_path,
                'evaluation_path': None,
                'status': 'benchmark_only',
                'error': str(e)
            }
    
    def get_benchmark_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive statistics from benchmark results.
        
        Args:
            df: DataFrame with benchmark results
            
        Returns:
            Dictionary with benchmark statistics
        """
        stats = {
            'total_queries': len(df),
            'successful_optimizations': len(df[df['optimization_status'] == 'success']),
            'failed_optimizations': len(df[df['optimization_status'] == 'failed']),
            'success_rate': len(df[df['optimization_status'] == 'success']) / len(df) * 100 if len(df) > 0 else 0
        }
        
        # Add evaluation statistics if available
        successful_outputs = df[df['optimization_status'] == 'success']['optimized_output'].tolist()
        if successful_outputs:
            eval_results = self.evaluator.evaluate_batch(successful_outputs)
            eval_df = pd.DataFrame([{
                'has_citation': m.has_citation,
                'has_statistic': m.has_statistic,
                'has_quote': m.has_quote,
                'structure_check': m.has_structure,
                'word_count': m.word_count
            } for m in eval_results])
            
            eval_stats = self.evaluator.calculate_summary_statistics(eval_df)
            stats.update(eval_stats)
        
        return stats
    
    def generate_benchmark_report(self, df: pd.DataFrame) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            df: DataFrame with benchmark results
            
        Returns:
            Formatted benchmark report
        """
        stats = self.get_benchmark_statistics(df)
        
        report = f"""
        GEO Optimization Benchmark Report
        ===============================
        
        Dataset: {self.dataset_name} ({self.dataset_split})
        
        Execution Statistics:
        -------------------
        Total Queries: {stats['total_queries']}
        Successful Optimizations: {stats['successful_optimizations']}
        Failed Optimizations: {stats['failed_optimizations']}
        Success Rate: {stats['success_rate']:.1f}%
        
        Content Quality (Successful Optimizations):
        ----------------------------------------
        Citation Presence: {stats.get('citation_presence_pct', 0):.1f}%
        Statistic Presence: {stats.get('statistic_presence_pct', 0):.1f}%
        Quote Presence: {stats.get('quote_presence_pct', 0):.1f}%
        Structure Presence: {stats.get('structure_presence_pct', 0):.1f}%
        Average Word Count: {stats.get('average_word_count', 0):.1f}
        
        Overall Performance Score: {(stats.get('citation_presence_pct', 0) + stats.get('statistic_presence_pct', 0) + stats.get('quote_presence_pct', 0) + stats.get('structure_presence_pct', 0)) / 4:.1f}%
        """
        
        return report


# Legacy function for backward compatibility
def sample_queries(dataset, n=10):
    """Legacy function wrapper for backward compatibility."""
    if isinstance(dataset, list):
        return random.sample(dataset, min(n, len(dataset)))
    return []


def benchmark_optimizer(n=10):
    """Legacy function wrapper for backward compatibility."""
    runner = BenchmarkRunner()
    df = runner.run_benchmark(n, save_results=False)
    return df.to_dict('records')


if __name__ == "__main__":
    """Main execution for standalone benchmark running."""
    try:
        runner = BenchmarkRunner()
        results = runner.run_full_benchmark_with_evaluation(n=10)
        
        if results['status'] == 'success':
            print(f"Benchmark results saved to {results['outputs_path']}")
            print(f"Evaluation report saved to {results['evaluation_path']}")
        else:
            print(f"Benchmark completed with status: {results['status']}")
            if 'error' in results:
                print(f"Error: {results['error']}")
                
    except Exception as e:
        print(f"Benchmark execution failed: {e}")
        sys.exit(1)
