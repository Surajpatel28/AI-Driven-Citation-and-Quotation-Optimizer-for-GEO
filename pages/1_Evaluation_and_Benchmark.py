"""
Evaluation and Benchmark Page

This page provides a clean interface for running benchmarks and evaluations
using the object-oriented architecture.
"""

import streamlit as st
import pandas as pd
import os
import subprocess
import sys

from core import BenchmarkRunner, EvaluationEngine


class EvaluationBenchmarkPage:
    """Streamlit page for evaluation and benchmarking."""
    
    def __init__(self):
        """Initialize the page."""
        self.benchmark_runner = BenchmarkRunner()
        self.evaluator = EvaluationEngine()
    
    def render_header(self):
        """Render the page header."""
        st.title("Evaluation & Benchmark")
        st.markdown("""
        This page lets you run the GEO-Bench benchmark and automatically evaluate 
        the outputs using the object-oriented architecture. Results and evaluation 
        statistics will be shown below.
        """)
    
    def run_benchmark_and_evaluation(self):
        """Run benchmark and evaluation."""
        with st.spinner("Running benchmark and evaluation..."):
            try:
                # Run benchmark with evaluation
                results = self.benchmark_runner.run_full_benchmark_with_evaluation(n=10)
                
                if results['status'] == 'success':
                    st.success("✅ Benchmark and evaluation completed successfully!")
                    st.info(f"📊 Results saved to: {results['outputs_path']}")
                    st.info(f"📋 Evaluation saved to: {results['evaluation_path']}")
                else:
                    st.warning(f"⚠️ Benchmark completed with status: {results['status']}")
                    if 'error' in results:
                        st.error(f"❌ Error: {results['error']}")
                
                return results
                
            except Exception as e:
                st.error(f"❌ Benchmark execution failed: {e}")
                return None
    
    def display_results(self):
        """Display benchmark and evaluation results."""
        output_path = os.path.join('result_from_benchmark', 'outputs.csv')
        eval_path = os.path.join('result_from_benchmark', 'eval_report.csv')
        
        if os.path.exists(output_path):
            st.subheader("📊 Latest Benchmark Outputs")
            df = pd.read_csv(output_path)
            st.dataframe(df.head())
            
            # Show success rate
            if 'optimization_status' in df.columns:
                success_rate = (df['optimization_status'] == 'success').mean() * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
        
        if os.path.exists(eval_path):
            st.subheader("📋 Evaluation Results")
            eval_df = pd.read_csv(eval_path)
            st.dataframe(eval_df)
            
            st.subheader("📈 Summary Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("Citation Presence (%)", f"{eval_df['has_citation'].mean() * 100:.1f}")
            col2.metric("Statistic Presence (%)", f"{eval_df['has_statistic'].mean() * 100:.1f}")
            col3.metric("Quote Presence (%)", f"{eval_df['has_quote'].mean() * 100:.1f}")
            col4.metric("Structure Presence (%)", f"{eval_df['structure_check'].mean() * 100:.1f}")
            col5.metric("Avg. Word Count", f"{eval_df['word_count'].mean():.1f}")
            
            st.subheader("📊 Feature Presence Distribution")
            feature_cols = ['has_citation', 'has_statistic', 'has_quote', 'structure_check']
            st.bar_chart(eval_df[feature_cols].mean())
            
            # Generate and display comprehensive report
            try:
                report = self.evaluator.generate_evaluation_report(eval_df)
                with st.expander("📄 Detailed Evaluation Report"):
                    st.text(report)
            except Exception as e:
                st.warning(f"Could not generate detailed report: {e}")
    
    def render(self):
        """Render the complete page."""
        self.render_header()
        
        # Button section
        if st.button("🚀 Run Benchmark & Evaluate", type="primary"):
            self.run_benchmark_and_evaluation()
        
        # Display results
        self.display_results()


def main():
    """Main function for the page."""
    page = EvaluationBenchmarkPage()
    page.render()


if __name__ == "__main__":
    main()
else:
    # When imported as a Streamlit page
    main()
