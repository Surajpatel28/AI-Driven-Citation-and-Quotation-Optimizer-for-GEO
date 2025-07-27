import re
import pandas as pd
import os

def has_citation(text):
    # Looks for URLs, 'Citation:', 'Source:', or numeric references like [1]
    return bool(re.search(r'(https?://|Citation:|Source:|\[\d+\])', text, re.IGNORECASE))

def has_statistic(text):
    # Looks for numbers with units, %, or 'according to'
    return bool(re.search(r'\b\d+[,.]?\d*\s?(%|percent|million|billion|thousand|[A-Za-z]+)?\b', text)) or 'according to' in text.lower()

def has_quote(text):
    # Looks for quoted text or attribution
    return bool(re.search(r'“[^”]+”|"[^"]+"', text)) or 'said' in text.lower() or 'according to' in text.lower()

def word_count(text):
    return len(text.split())

def structure_check(text):
    # Checks for headings, bullet points, numbered lists, section titles, or paragraph breaks
    patterns = [
        r'^\s*#{1,6}\s',                # Markdown headings
        r'^\s*[\-\*\•]\s',           # Bullet points
        r'^\s*\d+\.\s',               # Numbered lists
        r'\n\s*\n',                    # Paragraph breaks
        r'(?i)\b(introduction|conclusion|summary|background|results|discussion|references)\b'  # Section titles
    ]
    for pat in patterns:
        if re.search(pat, text, re.MULTILINE):
            return True
    return False

def evaluate_outputs(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    if 'optimized_output' not in df.columns:
        raise ValueError("CSV must have an 'optimized_output' column.")
    results = []
    for output in df['optimized_output']:
        results.append({
            'has_citation': has_citation(str(output)),
            'has_statistic': has_statistic(str(output)),
            'has_quote': has_quote(str(output)),
            'word_count': word_count(str(output)),
            'structure_check': structure_check(str(output))
        })
    results_df = pd.DataFrame(results)
    # Aggregate reporting
    print('Citation presence:', results_df['has_citation'].mean())
    print('Statistic presence:', results_df['has_statistic'].mean())
    print('Quote presence:', results_df['has_quote'].mean())
    print('Structure presence:', results_df['structure_check'].mean())
    print('Average word count:', results_df['word_count'].mean())
    # Save detailed report
    results_df.to_csv(output_path, index=False)
    print(f"Detailed evaluation report saved to {output_path}")

if __name__ == "__main__":
    # Example usage: expects a CSV with an 'optimized_output' column
    input_csv = os.path.join('result_from_benchmark', 'outputs.csv')
    output_csv = os.path.join('result_from_benchmark', 'eval_report.csv')
    evaluate_outputs(input_csv, output_csv)
