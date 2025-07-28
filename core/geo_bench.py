import sys
import os
import random
import pandas as pd
from datasets import load_dataset

# Always use absolute imports from core
try:
    from core.gemini_chain import optimize
except ImportError as e:
    print("[ERROR] Could not import from core.gemini_chain. Make sure you are running this script from the project root directory.")
    print(e)
    sys.exit(1)

# Check for required dependencies
try:
    import nltk
    import spacy
except ImportError as e:
    print("[ERROR] Required dependency not installed:", e)
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

# Load Geo-Bench dataset
ds = load_dataset("GEO-Optim/geo-bench", "test")
dataset_list = list(ds["test"])

# Sample a few queries for benchmarking
def sample_queries(dataset, n=10):
    return random.sample(dataset, n)

# Run optimizer and collect results
def benchmark_optimizer(n=10):
    queries = sample_queries(dataset_list, n)
    results = []
    for item in queries:
        query = item['query']
        tags = item.get('tags', [])
        sources = item.get('sources', [])
        optimized = optimize(query)
        results.append({
            'query': query,
            'tags': tags,
            'sources': sources,
            'optimized_output': optimized
        })
    return results

if __name__ == "__main__":
    results = benchmark_optimizer(n=10)
    df = pd.DataFrame(results)
    output_dir = 'result_from_benchmark'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'outputs.csv')
    df.to_csv(output_path, index=False)
    print(f"Benchmark results saved to {output_path}")

    # --- Automated Evaluation ---
    try:
        from evaluate_outputs import evaluate_outputs
        eval_output_path = os.path.join(output_dir, 'eval_report.csv')
        evaluate_outputs(output_path, eval_output_path)
        print(f"Evaluation report saved to {eval_output_path}")
    except Exception as e:
        print(f"Automated evaluation failed: {e}")
