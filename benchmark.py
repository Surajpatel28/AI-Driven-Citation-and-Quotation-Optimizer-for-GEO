"""
Object-Oriented Benchmark Script

This script provides object-oriented benchmarking functionality for the
GEO Citation Optimizer using the GEO-Bench dataset.
"""

import sys
from core import BenchmarkRunner


def main():
    """Main benchmark function."""
    try:
        print("Starting GEO Citation Optimizer Benchmark...")
        print("=" * 50)
        
        # Create benchmark runner
        runner = BenchmarkRunner()
        
        # Run full benchmark with evaluation
        results = runner.run_full_benchmark_with_evaluation(n=10)
        
        if results['status'] == 'success':
            print(f"✅ Benchmark completed successfully!")
            print(f"📊 Results saved to: {results['outputs_path']}")
            print(f"📋 Evaluation saved to: {results['evaluation_path']}")
        else:
            print(f"⚠️  Benchmark completed with status: {results['status']}")
            if 'error' in results:
                print(f"❌ Error: {results['error']}")
        
        print("=" * 50)
        print("Benchmark execution complete.")
        
    except Exception as e:
        print(f"❌ Benchmark execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
