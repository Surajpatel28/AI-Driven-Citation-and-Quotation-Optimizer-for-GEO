import streamlit as st
import pandas as pd
import os

st.title("Evaluation & Benchmark")

st.markdown("""
This page lets you run the GEO-Bench benchmark and automatically evaluate the outputs with a single click. Results and evaluation statistics will be shown below.
""")


if st.button("Run Benchmark and Evaluate"):
    with st.spinner("Running benchmark and evaluation. This may take a few minutes..."):
        import subprocess
        try:
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env = os.environ.copy()
            env["PYTHONPATH"] = project_root
            # Run geo_bench.py with the same Python executable as Streamlit
            bench_result = subprocess.run([
                sys.executable, os.path.join("core", "geo_bench.py")
            ], capture_output=True, text=True, cwd=project_root, env=env)
            st.code(bench_result.stdout)
            if bench_result.stderr:
                st.error(bench_result.stderr)
            # Run evaluate_outputs.py with the same Python executable
            eval_result = subprocess.run([
                sys.executable, "evaluate_outputs.py"
            ], capture_output=True, text=True, cwd=project_root, env=env)
            st.code(eval_result.stdout)
            if eval_result.stderr:
                st.error(eval_result.stderr)
            st.success("Benchmark and evaluation completed! See results below.")
        except Exception as e:
            st.error(f"Failed to run benchmark or evaluation: {e}")

# Show the latest outputs and evaluation summary
output_path = os.path.join('result_from_benchmark', 'outputs.csv')
eval_path = os.path.join('result_from_benchmark', 'eval_report.csv')
if os.path.exists(output_path):
    st.subheader("Latest Benchmark Outputs (first 5 rows)")
    df = pd.read_csv(output_path)
    st.dataframe(df.head())
if os.path.exists(eval_path):
    st.subheader("Evaluation Table")
    eval_df = pd.read_csv(eval_path)
    st.dataframe(eval_df)
    st.subheader("Summary Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Citation Presence (%)", f"{eval_df['has_citation'].mean() * 100:.1f}")
    col2.metric("Statistic Presence (%)", f"{eval_df['has_statistic'].mean() * 100:.1f}")
    col3.metric("Quote Presence (%)", f"{eval_df['has_quote'].mean() * 100:.1f}")
    col4.metric("Structure Presence (%)", f"{eval_df['structure_check'].mean() * 100:.1f}")
    col5.metric("Avg. Word Count", f"{eval_df['word_count'].mean():.1f}")
    st.subheader("Feature Presence Distribution")
    feature_cols = ['has_citation', 'has_statistic', 'has_quote', 'structure_check']
    st.bar_chart(eval_df[feature_cols].mean())
