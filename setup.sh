#!/bin/bash

# Post-installation script for Streamlit Cloud
# This script runs after pip install requirements.txt

echo "Installing spaCy language model..."
python -m spacy download en_core_web_sm

echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt')"

echo "Setup complete!"
