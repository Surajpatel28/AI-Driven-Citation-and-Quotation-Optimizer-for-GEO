# AI-Powered Citation & Quote Optimizer

## What is this?

This tool automatically improves your content by adding:
- **Authoritative citations** from reliable sources
- **Expert quotes** with proper attribution  
- **Relevant statistics** to support your claims

The goal is to make your content more trustworthy and visible to AI search engines like ChatGPT, Gemini, Claude, and Perplexity.

## Features

✨ **Smart Content Enhancement**
- Automatically adds citations, quotes, and statistics
- Makes content more credible and authoritative
- Optimized for AI search engines (GEO - Generative Engine Optimization)

🎯 **Easy to Use**
- Simple web interface built with Streamlit
- Professional custom styling with modern design
- Just paste your text and get optimized content
- Built-in evaluation to measure improvements

🔧 **For Developers**
- Clean object-oriented Python code
- Custom CSS styling system included
- Easy to extend and customize
- Comprehensive test suite included

## Quick Start

### 1. Get the Code

```bash
git clone https://github.com/Surajpatel28/AI-Driven-Citation-and-Quotation-Optimizer-for-GEO.git
cd AI-Driven-Citation-and-Quotation-Optimizer-for-GEO
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Your API Key

Create a `.env` file and add your Google Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

**Optional: Customize Performance Settings**

You can also customize various settings in your `.env` file:

```
# Performance settings
GEMINI_MODEL=gemini-2.5-flash     # Use faster model
BENCHMARK_SAMPLE_SIZE=5           # Smaller benchmark size
MAX_URL_CONTENT_LENGTH=3000       # Process shorter content

# Quality settings  
GEMINI_TEMPERATURE=0.3            # Lower = more focused output
TOP_SENTENCES_COUNT=3             # Number of key sentences to extract
```

See `.env.example` for all available options.

### 4. Run the App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start optimizing your content!

## How to Use

1. **Open the web app** in your browser
2. **Paste your content** into the text area
3. **Click "Optimize Content"** to enhance it with citations and quotes
4. **Review the results** and copy the improved content

## What Gets Added?

### Before Optimization:
```
Climate change is a serious issue that affects everyone.
```

### After Optimization:
```
Climate change is a serious issue that affects everyone. According to NASA, 
"Climate change refers to long-term shifts in temperatures and weather patterns" 
(NASA Climate Change, 2024). Recent studies show that global temperatures have 
risen by 1.1°C since pre-industrial times, with 97% of climate scientists 
agreeing that human activities are the primary cause (IPCC, 2023).
```

## Project Structure

```
📁 AI-Driven Citation and Quotation Optimizer/
├── 📄 app.py                      # Main web application
├── 📄 benchmark.py                # Run performance tests
├── 📄 evaluate_outputs.py         # Evaluate optimization quality
├── 📄 requirements.txt            # Required packages
├── 📄 test_oop.py                 # Test suite
├── 📁 core/                       # Main code modules
│   ├── 📄 content_processor.py   # Text processing
│   ├── 📄 geo_optimizer.py       # AI optimization engine
│   ├── 📄 evaluation_engine.py   # Quality evaluation
│   ├── 📄 benchmark_runner.py    # Performance testing
│   └── 📄 streamlit_app.py       # Web interface
└── 📁 result_from_benchmark/     # Test results
```

## For Developers

### Using the Code in Your Project

```python
from core import GEOOptimizer, ContentProcessor

# Initialize the optimizer
processor = ContentProcessor()
optimizer = GEOOptimizer()

# Optimize your content
content = "Your text here..."
optimized = optimizer.optimize_content(content)
print(optimized)
```

### Running Tests

```bash
python test_oop.py
```

### Running Benchmarks

```bash
python benchmark.py
```

## Performance Metrics

The tool tracks several quality metrics:
- **Citation Rate**: Percentage of content with authoritative sources
- **Quote Integration**: Expert quotes with proper attribution
- **Statistical Support**: Relevant statistics and data
- **Structure Quality**: Proper headings and organization

## Customization

You can easily customize the optimization by:
- Modifying prompts in `core/geo_optimizer.py`
- Adding new evaluation metrics in `core/evaluation_engine.py`
- Extending the content processor for specialized domains

## Requirements

- Python 3.8+
- Google Gemini API key
- Internet connection for AI processing

## Limitations

- Requires a valid Gemini API key
- Processing speed depends on content length
- AI-generated citations should be fact-checked
- Best results with factual, informational content

## Research Background

This project is based on the GEO (Generative Engine Optimization) research framework, which studies how to optimize content for AI-powered search engines. The approach follows E-E-A-T principles (Experience, Expertise, Authoritativeness, Trustworthiness) to improve content credibility.

## Citation

If you use this project in your research, please cite:

```
@inproceedings{10.1145/3637528.3671900,
author = {Aggarwal, Pranjal and Murahari, Vishvak and Rajpurohit, Tanmay and Kalyan, Ashwin and Narasimhan, Karthik and Deshpande, Ameet},
title = {GEO: Generative Engine Optimization},
year = {2024},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3637528.3671900},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {5–16}
}
```

## License

This project is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
