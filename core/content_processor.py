"""
Content Processing Module for GEO Citation Optimizer

This module handles all text processing operations including:
- URL content extraction
- Content preprocessing and cleaning
- Text classification
- Entity extraction and analysis
"""

import re
import sys
import os
import time
import logging
import tempfile
import requests
import nltk
import spacy
import numpy as np
import subprocess
from typing import Dict, List, Optional, Union
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import subprocess
import sys

class ContentProcessor:
    """
    A comprehensive content processing engine for the GEO Citation Optimizer.
    
    This class handles all aspects of content processing from URL extraction
    to advanced text analysis and preprocessing.
    """
    
    def __init__(self, 
                 sentence_model: str = "all-MiniLM-L6-v2",
                 spacy_model: str = "en_core_web_sm",
                 tokenizer_model: str = "google/flan-t5-large"):
        """
        Initialize the ContentProcessor with required NLP models.
        
        Args:
            sentence_model: SentenceTransformer model for embeddings
            spacy_model: spaCy model for NER
            tokenizer_model: Tokenizer model for token estimation
        """
        self.logger = self._setup_logging()
        self._initialize_nltk()
        self._initialize_spacy(spacy_model)
        self._initialize_sentence_transformer(sentence_model)
        self._initialize_tokenizer(tokenizer_model)
        
        # Precompiled patterns for better performance
        self._compile_patterns()
    
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
    
    def _initialize_nltk(self) -> None:
        """Initialize NLTK data."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:  # Raised when resource is not found
            self.logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt")

    def _initialize_spacy(self, model_name: str) -> None:
        """Initialize spaCy model with fallback for deployment environments."""
        try:
            self.nlp = spacy.load(model_name)
            self.logger.info(f"Successfully loaded spaCy model: {model_name}")
        except OSError:  # Raised when model is not installed
            # Try to download only in local development environment
            if self._is_local_environment():
                self.logger.info(f"Downloading spaCy model '{model_name}'...")
                try:
                    result = subprocess.run([sys.executable, "-m", "spacy", "download", model_name], 
                                          capture_output=True, text=True, check=True)
                    self.logger.info(f"Successfully downloaded {model_name}")
                    self.nlp = spacy.load(model_name)
                    return
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Failed to download spaCy model: {e}")
            
            # Fallback: disable spaCy features gracefully
            self.logger.warning(f"spaCy model '{model_name}' not available. NER features will be disabled.")
            self.nlp = None
    
    def _is_local_environment(self) -> bool:
        """Check if running in local development environment."""
        # Check for common deployment environment indicators
        deployment_indicators = [
            'STREAMLIT_CLOUD',
            'HEROKU',
            'RAILWAY',
            'VERCEL',
            'RENDER',
            'PYTHONPATH'  # Often set in containerized environments
        ]
        
        for indicator in deployment_indicators:
            if indicator in os.environ:
                return False
                
        # Check if we can write to the Python environment (local dev usually can)
        try:
            with tempfile.NamedTemporaryFile():
                return True
        except:
            return False
    
    def _initialize_sentence_transformer(self, model_name: str) -> None:
        """Initialize SentenceTransformer model."""
        try:
            self.sentence_model = SentenceTransformer(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer: {e}")
            self.sentence_model = None
    
    def _initialize_tokenizer(self, model_name: str) -> None:
        """Initialize tokenizer for token estimation."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            self.tokenizer = None
    
    def _compile_patterns(self) -> None:
        """Precompile regex patterns for better performance."""
        # URL pattern
        self.url_pattern = re.compile(
            r'^(https?://)?(www\.)?([A-Za-z_0-9-]+)+(\.[a-z]{2,})+(/[^\s]*)?$'
        )
        
        # Query pattern
        query_keywords = [
            "what", "why", "how", "when", "where", "who", "which", "whom", "whose",
            "is", "are", "can", "do", "does", "should", "tell me", "explain", "give me"
        ]
        self.query_pattern = re.compile(
            r"^(?:" + "|".join(re.escape(q) for q in query_keywords) + r")\b", 
            re.IGNORECASE
        )
        
        # Boilerplate patterns
        self.boilerplate_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in [
                r'copyright \d{4}', r'all rights reserved', r'cookie policy', 
                r'privacy policy', r'\bsubscribe\b', r'\blogin\b', 
                r'\bsign up\b', r'\bterms of service\b'
            ]
        ]
        
        # Statistics patterns
        self.stat_patterns = [
            re.compile(
                r'\d+(?:\.\d+)?\s*(?:%|percent|million|billion|trillion|years?|people|cases|deaths|dollars|euros|pounds|students|citizens|votes|km|miles|liters|tons|hectares|acres)',
                re.IGNORECASE
            )
        ]
        
        # Weak claim patterns
        weak_pattern_strings = [
            r"\bstudies show\b", r"\bsome people say\b", r"\bmany believe\b",
            r"\boften\b", r"\balways\b", r"\bnever\b", r"\bexperts claim\b",
            r"\bis the best\b", r"\bnumber one\b", 
            r"\baccording to (?!\w+ \([\d]{4}\))", 
            r"\b\d+%.*(prefer|agree|support|believe)\b",
            r"it is believed that", r"researchers suggest", 
            r"recent studies", r"it is said that"
        ]
        self.weak_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in weak_pattern_strings
        ]
    
    def is_url(self, text: str) -> bool:
        """
        Check if the given text is a valid URL.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is a URL, False otherwise
        """
        return bool(self.url_pattern.match(text.strip()))
    
    def fetch_text_from_url(self, url: str, max_length: int = 5000) -> str:
        """
        Extract text content from a URL.
        
        Args:
            url: URL to extract content from
            max_length: Maximum length of extracted text
            
        Returns:
            Extracted text content
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = "\n\n".join([p.get_text() for p in paragraphs])
            
            return text.strip()[:max_length]
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP Request failed for URL {url}: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to extract content from URL {url}: {e}")
            return ""
    
    def classify_input_type(self, text: str) -> str:
        """
        Classify input text as query, blurb, or article.
        
        Args:
            text: Input text to classify
            
        Returns:
            Classification: 'query', 'blurb', or 'article'
        """
        text = text.strip()
        word_count = len(text.split())
        
        if text.endswith("?") and self.query_pattern.match(text):
            return "query"
        elif word_count < 40:
            return "blurb"
        else:
            return "article"
    
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for the given text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        if self.tokenizer is None:
            # Fallback: rough estimation
            return int(len(text.split()) * 1.3)
        
        try:
            return len(self.tokenizer.encode(text, truncation=False))
        except Exception as e:
            self.logger.error(f"Token estimation failed: {e}")
            return int(len(text.split()) * 1.3)
    
    def get_max_output_tokens(self, text: str, content_type: str = "article") -> int:
        """
        Calculate maximum output tokens based on input and content type.
        
        Args:
            text: Input text
            content_type: Type of content ('query', 'blurb', 'article')
            
        Returns:
            Maximum output tokens
        """
        input_tokens = self.estimate_token_count(text)
        
        # Use adaptive multipliers per content type
        if content_type == "query":
            return min(512, 256 + input_tokens)
        elif content_type == "blurb":
            return min(int(input_tokens * 1.25), 2048)
        elif content_type == "article":
            return min(int(input_tokens * 1.5), 8192)
        else:
            return min(int(input_tokens * 1.2), 1024)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Normalize whitespace
        clean = re.sub(r'\s+', ' ', text).strip()
        
        # Remove boilerplate content
        for pattern in self.boilerplate_patterns:
            clean = pattern.sub('', clean)
        
        return clean
    
    def _extract_sentences(self, text: str) -> List[str]:
        """
        Extract and deduplicate sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of unique sentences
        """
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            self.logger.warning(f"NLTK sentence split failed: {e}")
            sentences = [text]
        
        # Deduplicate and filter
        seen = set()
        filtered = []
        
        for sent in sentences:
            s_clean = sent.strip().lower()
            if len(s_clean) < 8 or s_clean in seen:
                continue
            seen.add(s_clean)
            filtered.append(sent.strip())
        
        return filtered
    
    def _extract_entities(self, text: str) -> List[tuple]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity_text, entity_label) tuples
        """
        if self.nlp is None:
            self.logger.warning("spaCy model not available, skipping entity extraction")
            return []
            
        try:
            doc = self.nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        except Exception as e:
            self.logger.warning(f"spaCy NER failed: {e}")
            return []
    
    def _extract_statistics(self, text: str) -> List[str]:
        """
        Extract statistics and numerical data from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted statistics
        """
        stats = []
        for pattern in self.stat_patterns:
            stats.extend(pattern.findall(text))
        return stats
    
    def _identify_weak_sentences(self, sentences: List[str]) -> List[str]:
        """
        Identify sentences with weak claims that need strengthening.
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            List of sentences with weak claims
        """
        weak_sentences = []
        
        for sent in sentences:
            for pattern in self.weak_patterns:
                if pattern.search(sent):
                    weak_sentences.append(sent)
                    break
        
        return weak_sentences
    
    def _get_top_sentences(self, sentences: List[str], top_k: int = 3) -> List[str]:
        """
        Get most important sentences using embedding-based centrality.
        
        Args:
            sentences: List of sentences
            top_k: Number of top sentences to return
            
        Returns:
            List of top sentences
        """
        if not sentences or self.sentence_model is None:
            return []
        
        try:
            # Batch processing for efficiency
            batch_size = 16
            embeddings = []
            
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                try:
                    batch_emb = self.sentence_model.encode(batch)
                    embeddings.extend(batch_emb)
                except Exception as e:
                    self.logger.warning(f"Embedding batch failed: {e}")
            
            if not embeddings:
                return []
            
            embeddings = np.array(embeddings)
            
            # Calculate centrality based on similarity to all other sentences
            sim_matrix = np.inner(embeddings, embeddings)
            avg_sim = sim_matrix.mean(axis=1)
            top_idx = np.argsort(avg_sim)[-top_k:]
            
            return [sentences[i] for i in sorted(top_idx)]
            
        except Exception as e:
            self.logger.warning(f"Top sentence selection failed: {e}")
            return []
    
    def preprocess_content(self, text: str, fast_mode: bool = False) -> Dict[str, Union[str, List]]:
        """
        Comprehensive content preprocessing with timing information.
        
        Args:
            text: Raw input text
            fast_mode: If True, skip expensive operations for speed
            
        Returns:
            Dictionary containing processed content components
        """
        start_time = time.time()
        
        # 1. Text cleaning
        t0 = time.time()
        clean_text = self._clean_text(text)
        self.logger.debug(f"[Timing] Cleaning: {time.time() - t0:.3f}s")
        
        if not clean_text:
            return {
                "clean_text": "",
                "sentences": [],
                "entities": [],
                "stats": [],
                "top_sentences": [],
                "weak_sentences": []
            }
        
        # 2. Sentence extraction and deduplication
        t1 = time.time()
        sentences = self._extract_sentences(clean_text)
        self.logger.debug(f"[Timing] Sentence extraction: {time.time() - t1:.3f}s")
        
        if fast_mode:
            # In fast mode, skip expensive operations
            total_time = time.time() - start_time
            self.logger.info(f"[Timing] Total preprocessing (fast): {total_time:.3f}s")
            
            return {
                "clean_text": clean_text,
                "sentences": sentences[:5],  # Limit to first 5 sentences
                "entities": [],
                "stats": [],
                "top_sentences": sentences[:3],
                "weak_sentences": []
            }
        
        # 3. Entity extraction
        t2 = time.time()
        entities = self._extract_entities(clean_text)
        self.logger.debug(f"[Timing] Entity extraction: {time.time() - t2:.3f}s")
        
        # 4. Statistics extraction
        t3 = time.time()
        stats = self._extract_statistics(clean_text)
        self.logger.debug(f"[Timing] Statistics extraction: {time.time() - t3:.3f}s")
        
        # 5. Weak claim detection
        t4 = time.time()
        weak_sentences = self._identify_weak_sentences(sentences)
        self.logger.debug(f"[Timing] Weak claim detection: {time.time() - t4:.3f}s")
        
        # 6. Top sentence selection
        t5 = time.time()
        top_sentences = self._get_top_sentences(sentences)
        self.logger.debug(f"[Timing] Top sentence selection: {time.time() - t5:.3f}s")
        
        total_time = time.time() - start_time
        self.logger.info(f"[Timing] Total preprocessing: {total_time:.3f}s")
        
        return {
            "clean_text": clean_text,
            "sentences": sentences,
            "entities": entities,
            "stats": stats,
            "top_sentences": top_sentences,
            "weak_sentences": weak_sentences
        }
