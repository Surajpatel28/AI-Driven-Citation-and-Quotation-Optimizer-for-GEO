"""
GEO Optimizer Module

This module contains the main optimization engine that uses Gemini AI
to enhance content with citations, quotations, and statistics for better
Generative Engine Optimization (GEO).
"""

import os
import time
import logging
from typing import Dict, Optional
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from .content_processor import ContentProcessor


class GEOOptimizer:
    """
    Main optimization engine for the GEO Citation Optimizer.
    
    This class handles content optimization using Gemini AI to inject
    authoritative citations, expert quotations, and relevant statistics
    following E-E-A-T principles.
    """
    
    def __init__(self,
                 model_name: str = "gemini-2.5-flash",
                 temperature: float = 0.3,
                 content_processor: Optional[ContentProcessor] = None):
        self.logger = self._setup_logging()
        self._load_environment()
        self._initialize_llm(model_name, temperature)
        self.content_processor = content_processor or ContentProcessor()
        self._build_prompt_templates()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        # Setup file logging for errors
        log_path = os.path.join(os.path.dirname(__file__), '..', 'error.log')
        logging.basicConfig(
            filename=log_path,
            level=logging.ERROR,
            format='%(asctime)s %(levelname)s %(message)s'
        )
        
        # Also setup console logging
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_environment(self) -> None:
        """Load environment variables."""
        load_dotenv()
    
    def _initialize_llm(self, model_name: str, temperature: float) -> None:
        """Initialize the Gemini LLM with optimized settings."""
        try:
            # Import safety settings for 2.5 Flash
            from langchain_google_genai import HarmBlockThreshold, HarmCategory
            
            # Optimize settings for Gemini 2.5 Flash
            if "2.5" in model_name:
                # Gemini 2.5 Flash optimized settings with thinking support
                self.llm = GoogleGenerativeAI(
                    model=model_name, 
                    temperature=temperature,
                    # max_output_tokens=2048,  # Increased to handle thinking tokens
                    top_k=40,               # Better for 2.5 Flash
                    top_p=0.95,             # Higher diversity for better results
                    # Add safety settings to prevent blocking
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                self.logger.info(f"Initialized Gemini 2.5 Flash with thinking support")
            else:
                # Legacy settings for 1.5 Flash
                self.llm = GoogleGenerativeAI(
                    model=model_name, 
                    temperature=temperature,
                    # max_output_tokens=512,  # Original limit for 1.5 Flash
                    top_k=1,               # More focused responses
                    top_p=0.8              # Balanced creativity vs speed
                )
            self.logger.info(f"Initialized Gemini model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _build_prompt_templates(self) -> None:
        """Build simplified, essential prompt templates."""

        # Main optimization template
        self.optimization_template = """
You are a specialized Generative Engine Optimization (GEO) content editor. Your only job is to enhance the provided content so it is more credible, engaging, and valuable for generative search results. Follow these instructions exactly:

1. Add at least one authoritative citation, one expert quotation, and one relevant statistic — all from credible, verifiable sources published within the last 5 years (2019 or later). Only use older data if no newer credible data exists, and clearly indicate the year. Integrate them naturally into the body of the text.
2. Citations from credible primary sources, and formatted as inline Markdown hyperlinks with source name + year in the sentence.
2. Apply the E-E-A-T framework (Experience, Expertise, Authoritativeness, Trustworthiness) throughout.
3. Do not:
   - Invent people, experts, or studies.
   - Use placeholders like `[Insert...]` or `(Insert...)`.
   - Include fabricated names (e.g., `Dr. [Name]`) or unverifiable study references.
   - Use vague or unsupported percentages such as “research shows that 60%...”.
4. Do not use placeholders such as “citation needed” or “quote here”.
5. Keep the original meaning intact while improving clarity, depth, and structure.
6. Focus on strengthening any weak or unsupported claims with authoritative backing.
7. Output only the final enhanced Markdown content. Do not include explanations, reasoning, notes, or any text outside the enhanced content.
8. Do not exceed 250 words in the output.

---

Content to enhance (pre-cleaned and optimized for processing):
{content}

---

Final Enhanced Content (Markdown only, no commentary):
"""

        # Query optimization template
        self.query_template = PromptTemplate(
            input_variables=["content"],
            template="""
            Enhance this query response with professional clarity and structure.
            Use only verifiable information. Keep under 150 words, professional tone.
            If you mention any citations, quotes or stats, ensure they are well-established and verifiable.

            Query: {content}

            Enhanced response:
            """
        )

        # AI simulation template
        self.simulation_template = """
        You are an AI assistant answering this user query:
        User Query: "{user_query}"
        
        Use the content below as context:
        Content: {content}
        
        Provide a helpful response using the content provided. Be factual and cite relevant parts.
        """

    def _sanitize_output(self, text: str) -> str:
        """Remove placeholders & ensure clickable URLs."""
        import re

        # Log original text length
        self.logger.debug(f"Sanitizing text with {len(text)} characters")

        # If placeholders exist, return empty to trigger regeneration
        if "[Insert" in text or "(Insert" in text:
            self.logger.warning("Found placeholders in text, returning empty")
            return ""

        # Fix malformed markdown links first
        # Pattern: ([url](url)] -> [url](url)
        text = re.sub(r'\(\[([^\]]+)\]\(([^)]+)\)\]?\)', r'[\1](\2)', text)
        
        # Pattern: (url) that's not part of markdown -> leave as is or convert
        # Only convert standalone URLs that aren't already in markdown format
        # Negative lookbehind (?<!\]\() means "not preceded by ]("
        # Negative lookahead (?![^[]*\]) means "not followed by ] without [ in between"
        def convert_standalone_url(match):
            url = match.group(0)
            return f"[{url}]({url})"
        
        # Match URLs that are NOT already in markdown links
        text = re.sub(r'(?<!\]\()(?<![\[\(])https?://[^\s\)]+(?![^[]*\])', convert_standalone_url, text)
        
        self.logger.debug(f"Sanitized text has {len(text)} characters")
        return text    
    
    def optimize_content(self, raw_text: str, fast_mode: bool = True) -> str:
        """Optimize content using Gemini AI."""
        start_time = time.time()
        try:
            self.logger.info(f"Starting optimization with {len(raw_text)} characters")
            content_type = self.content_processor.classify_input_type(raw_text)

            if content_type == "query":
                return self._optimize_query(raw_text)

            preprocessed = self.content_processor.preprocess_content(raw_text, fast_mode=fast_mode)
            
            # Extract clean text and insights from preprocessing
            cleaned_content = preprocessed.get('clean_text', raw_text)
            weak_sentences = preprocessed.get('weak_sentences', [])
            entities = preprocessed.get('entities', [])
            
            # Use the optimization template with cleaned content
            prompt = self.optimization_template.format(content=cleaned_content)
            self.logger.info("Sending request to Gemini...")
            self.logger.debug(f"Using cleaned content ({len(cleaned_content)} chars vs {len(raw_text)} original chars)")
            
            # Fix for Gemini 2.5 Flash - handle thinking tokens and proper response
            try:
                response = self.llm.invoke(prompt)
                
                # Handle different response types from Gemini 2.5 Flash
                if isinstance(response, str):
                    optimized_content = response
                elif hasattr(response, 'content') and response.content:
                    optimized_content = response.content
                elif hasattr(response, 'text') and response.text:
                    optimized_content = response.text
                else:
                    optimized_content = str(response)
                
                optimized_content = optimized_content.strip()
                self.logger.info(f"Received response with {len(optimized_content)} characters")
                
                # Check if response is empty (common with 2.5 Flash MAX_TOKENS issue)
                if not optimized_content:
                    self.logger.warning("Empty response from Gemini, returning original content")
                    return raw_text
                    
            except Exception as e:
                self.logger.error(f"Error in Gemini invoke: {str(e)}")
                return raw_text

            # Final sanitization
            sanitized_content = self._sanitize_output(optimized_content)
            self.logger.info(f"After sanitization: {len(sanitized_content)} characters")

            self.logger.info(f"[Timing] Total optimization: {time.time() - start_time:.3f}s")
            
            # Return optimized content or original if sanitization failed
            final_result = sanitized_content or raw_text
            self.logger.info(f"Final result: {len(final_result)} characters")
            return final_result

        except Exception as e:
            self.logger.error(f"Error in optimization: {e}", exc_info=True)
            return raw_text  # Return original if optimization fails

    def _optimize_query(self, query: str) -> str:
        """
        Optimize a query-type input.
        
        Args:
            query: Query string to optimize
            
        Returns:
            Optimized query response
        """
        query_chain = self.query_template | self.llm
        start_time = time.time()
        
        try:
            optimized_response = query_chain.invoke({"content": query})
            self.logger.debug(f"[Timing] LLM Query invoke: {time.time() - start_time:.3f}s")
            return optimized_response
        except Exception as e:
            self.logger.error(f"LLM invocation failed for query: {e}", exc_info=True)
            return "Sorry, we encountered an error while optimizing this content for your query. Please try again later."
    
    def simulate_ai_response(self, content: str, user_query: str) -> str:
        """
        Simulate an AI response to a user query using the provided content.
        
        Args:
            content: Source content to use for response
            user_query: User's query
            
        Returns:
            Simulated AI response
        """
        prompt = self.simulation_template.format(
            user_query=user_query,
            content=content
        )
        
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            self.logger.error(f"LLM invocation failed in simulate_ai_response: {e}", exc_info=True)
            return "Sorry, we encountered an error while generating the AI response. Please try again later."
    
    def generate_answer_for_query(self, query: str) -> str:
        """
        Generate an optimized answer for a direct query.
        
        Args:
            query: Query to answer
            
        Returns:
            Optimized answer
        """
        return self._optimize_query(query)
    
    def get_optimization_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the optimization process.
        
        Returns:
            Dictionary with optimization statistics
        """
        # This could be extended to track optimization metrics
        return {
            "total_optimizations": getattr(self, '_optimization_count', 0),
            "successful_optimizations": getattr(self, '_successful_optimizations', 0),
            "failed_optimizations": getattr(self, '_failed_optimizations', 0)
        }


# Legacy function wrappers for backward compatibility
def optimize(raw_text: str) -> str:
    """
    Legacy function wrapper for backward compatibility.
    
    Args:
        raw_text: Raw text to optimize
        
    Returns:
        Optimized content
    """
    optimizer = GEOOptimizer()
    return optimizer.optimize_content(raw_text)


def simulate_ai_response(content: str, user_query: str) -> str:
    """
    Legacy function wrapper for backward compatibility.
    
    Args:
        content: Source content
        user_query: User query
        
    Returns:
        Simulated AI response
    """
    optimizer = GEOOptimizer()
    return optimizer.simulate_ai_response(content, user_query)


def generate_answer_for_query(query: str) -> str:
    """
    Legacy function wrapper for backward compatibility.
    
    Args:
        query: Query to answer
        
    Returns:
        Generated answer
    """
    optimizer = GEOOptimizer()
    return optimizer.generate_answer_for_query(query)
