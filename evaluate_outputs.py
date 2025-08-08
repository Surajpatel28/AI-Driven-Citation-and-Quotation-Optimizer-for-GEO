"""
Object-Oriented Evaluation Script

This script provides object-oriented evaluation functionality for the
GEO Citation Optimizer outputs.
"""

import os
import sys
from core import EvaluationEngine


def main():
    """Main evaluation function."""
    # Set up paths
    input_csv = os.path.join('result_from_benchmark', 'outputs.csv')
    output_csv = os.path.join('result_from_benchmark', 'eval_report.csv')
    
    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
        print("Please run the benchmark first to generate outputs.")
        sys.exit(1)
    
    try:
        # Create evaluator and run evaluation
        evaluator = EvaluationEngine()
        evaluator.evaluate_outputs_from_file(input_csv, output_csv)
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
