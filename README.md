# AI-Driven Citation & Quotation Optimizer for GEO

## Overview

This project is a **Proof of Concept (PoC)** for an AI-powered content optimizer that automatically injects authoritative citations, expert quotations, and relevant statistics into text. The goal is to enhance content credibility and visibility for generative engines (like ChatGPT, Gemini, Claude, and Perplexity) by aligning with the latest **Generative Engine Optimization (GEO)** and **E-E-A-T** (Experience, Expertise, Authoritativeness, Trustworthiness) principles.

Inspired by recent research and the 2025 GEO framework updates, this tool analyzes input content, identifies gaps in credibility signals, and rewrites sections to maximize trust and search performance.

---

## Features

- **Automatic Citation Injection:** Adds authoritative sources and references.
- **Expert Quotation Integration:** Inserts relevant expert quotes with attribution.
- **Statistic Enrichment:** Includes up-to-date statistics to support claims.
- **E-E-A-T Optimization:** Enhances Experience, Expertise, Authoritativeness, and Trustworthiness signals.
- **Streamlit Frontend:** User-friendly web interface for input, optimization, and evaluation.
- **Automated Benchmarking & Evaluation:** One-click benchmarking and detailed metrics (citation, quote, statistic, structure presence, etc.).
- **Extensible Evaluation Pipeline:** Easily add new automated checks or metrics.

---

## Research & References

- **GEO Framework:** [GEO-Optim/geo-bench](https://huggingface.co/datasets/GEO-Optim/geo-bench)
- **E-E-A-T Principles:** [Google Search Quality Rater Guidelines](https://static.googleusercontent.com/media/guidelines.raterhub.com/en//searchqualityevaluatorguidelines.pdf)
- **Prompt Engineering:** [OpenAI Cookbook: Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- **Citation/Fact-Checking:** [Cohan et al., 2019, ACL Anthology](https://aclanthology.org/N19-1371/), [Thorne et al., 2018, FEVER](https://aclanthology.org/N18-1074/)
- **Chain-of-Thought Prompting:** [Wei et al., 2022](https://arxiv.org/abs/2201.11903)

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/geo-citation-optimizer.git
cd geo-citation-optimizer
```

### 2. Install Dependencies

```bash
# (Recommended) Create a virtual environment
python -m venv geo-optimizer
source geo-optimizer/bin/activate  # On Windows: geo-optimizer\Scripts\activate

pip install -r requirements.txt
```

### 3. Download NLTK/Spacy Data (if needed)

```bash
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

### 5. Benchmark & Evaluate

- Use the **Evaluation & Benchmark** page in the app to run the full pipeline and view results.
- Outputs and evaluation reports are saved in `result_from_benchmark/`.

---

## Project Structure

```
core/
    gemini_chain.py      # Main optimization logic
    utils.py             # Preprocessing and helper functions
    geo_bench.py         # Benchmark runner
evaluate_outputs.py      # Automated evaluation script
app.py                   # Streamlit frontend
pages/
    1_Evaluation_and_Benchmark.py  # Combined evaluation/benchmark UI
result_from_benchmark/
    outputs.csv
    eval_report.csv
```

---

## Evaluation Metrics

- **Citation Presence:** % of outputs with at least one citation.
- **Statistic Presence:** % with at least one statistic.
- **Quote Presence:** % with at least one expert quote.
- **Structure Presence:** % with headings, lists, or sections.
- **Average Word Count:** Output length.
- **Detailed Table:** Per-output feature flags for manual review.

---

## Customization & Extensibility

- **Prompt Engineering:** Edit `core/gemini_chain.py` to refine prompt instructions or add few-shot examples.
- **Evaluation:** Modify `evaluate_outputs.py` to add new checks or metrics.
- **Datasets:** Swap in your own content or use the included GEO-Bench dataset.

---

## Limitations & Future Work

- **Human Evaluation:** For production, add human-in-the-loop review for quality assurance.

---


## Citation

If you use this project or the GEO-Bench dataset in your work, please cite:

```
@inproceedings{10.1145/3637528.3671900,
author = {Aggarwal, Pranjal and Murahari, Vishvak and Rajpurohit, Tanmay and Kalyan, Ashwin and Narasimhan, Karthik and Deshpande, Ameet},
title = {GEO: Generative Engine Optimization},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3637528.3671900},
doi = {10.1145/3637528.3671900},
abstract = {The advent of large language models (LLMs) has ushered in a new paradigm of search engines that use generative models to gather and summarize information to answer user queries. This emerging technology, which we formalize under the unified framework of generative engines (GEs), can generate accurate and personalized responses, rapidly replacing traditional search engines like Google and Bing. Generative Engines typically satisfy queries by synthesizing information from multiple sources and summarizing them using LLMs. While this shift significantly improvesuser utility and generative search engine traffic, it poses a huge challenge for the third stakeholder -- website and content creators. Given the black-box and fast-moving nature of generative engines, content creators have little to no control over when and how their content is displayed. With generative engines here to stay, we must ensure the creator economy is not disadvantaged. To address this, we introduce Generative Engine Optimization (GEO), the first novel paradigm to aid content creators in improving their content visibility in generative engine responses through a flexible black-box optimization framework for optimizing and defining visibility metrics. We facilitate systematic evaluation by introducing GEO-bench, a large-scale benchmark of diverse user queries across multiple domains, along with relevant web sources to answer these queries. Through rigorous evaluation, we demonstrate that GEO can boost visibility by up to 40\% in generative engine responses. Moreover, we show the efficacy of these strategies varies across domains, underscoring the need for domain-specific optimization methods. Our work opens a new frontier in information discovery systems, with profound implications for both developers of generative engines and content creators.},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {5–16},
numpages = {12},
keywords = {datasets and benchmarks, generative models, search engines},
location = {Barcelona, Spain},
series = {KDD '24'}
}
```

---

## License

[MIT License](LICENSE)

---

## Contact

For questions or collaboration, please contact [suraj123patel123@gmail.com](mailto:suraj123patel123@gmail.com).
