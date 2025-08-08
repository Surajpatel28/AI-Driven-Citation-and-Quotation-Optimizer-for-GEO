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
                 model_name: str = "gemini-1.5-flash",  # Use faster model
                 temperature: float = 0.2,               # Lower temp for faster generation
                 content_processor: Optional[ContentProcessor] = None):
        """
        Initialize the GEO Optimizer.
        
        Args:
            model_name: Gemini model to use (default: faster gemini-1.5-flash)
            temperature: Model temperature for generation (lower = faster)
            content_processor: ContentProcessor instance (creates new if None)
        """
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
            # Use faster model configuration
            self.llm = GoogleGenerativeAI(
                model=model_name, 
                temperature=temperature,
                max_output_tokens=512,  # Limit output for faster response
                top_k=1,               # More focused responses
                top_p=0.8              # Balanced creativity vs speed
            )
            self.logger.info(f"Initialized Gemini model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _build_prompt_templates(self) -> None:
        """Build prompt templates for different optimization scenarios."""
        # Main optimization prompt template (optimized)
        self.optimization_instructions = (
            "REQUIRED: 1 citation + 1 expert quote + 1 statistic\\n\\n"
            "You are a GEO content editor. Enhance the following content using E-E-A-T principles "
            "(Experience, Expertise, Authoritativeness, Trustworthiness). "
            "Add authoritative citations, expert quotes, and relevant statistics while preserving "
            "the original meaning and tone.\\n"
        )
        
        self.enhancement_instructions = """
        Enhancement Guidelines:
        1. Add one expert quote with attribution
        2. Include one relevant statistic with source  
        3. Add one authoritative citation
        4. Keep structure clear and professional
        """
        
        self.output_format = (
            "Output Format:\\n"
            "Return only the enhanced content. No explanations. Under 400 words.\\n"
        )
        
        # Query-specific prompt template (optimized for speed)
        self.query_template = PromptTemplate(
            input_variables=["content"],
            template="""
            Enhance this query response with E-E-A-T principles. Include:
            1. One citation (URL/source)
            2. One expert quote  
            3. One statistic
            
            Query: {content}
            
            Response (under 150 words, professional tone):
            """
        )
        
        # Simulation prompt for AI responses
        self.simulation_template = """
        You are an AI assistant answering the following user query:

        User Query: "{user_query}"

        Use the content below to generate your response.
        Content:
        {content}

        Only use facts from the provided content. Be helpful and cite relevant parts clearly.
        """
    
    def build_optimization_prompt(self, raw_text: str, preprocessed: Dict) -> str:
        """
        Build a dynamic prompt for content optimization.
        
        Args:
            raw_text: Original raw text
            preprocessed: Preprocessed content dictionary
            
        Returns:
            Complete optimization prompt
        """
        # Content section
        sentences = " ".join(preprocessed["sentences"])
        content_section = f"Content to Optimize:\\n{sentences}\\n"
        
        # Context section (only include if available)
        context_parts = []
        
        if preprocessed.get("top_sentences"):
            top_clips = "\\n".join(preprocessed["top_sentences"])
            if top_clips.strip():
                context_parts.append(f"Top Sentences to Emphasize:\\n{top_clips}")
        
        if preprocessed.get("weak_sentences"):
            weak_claims = " ".join(preprocessed["weak_sentences"])
            if weak_claims.strip():
                context_parts.append(f"Flagged Weak Claims (for strengthening or citation):\\n{weak_claims}")
        
        if preprocessed.get("stats"):
            stats = ", ".join(preprocessed["stats"])
            if stats.strip():
                context_parts.append(f"Relevant Statistics Found:\\n{stats}")
        
        context_section = "\\n\\n".join(context_parts)
        if context_section:
            context_section = f"Context to Guide Enhancement:\\n{context_section}\\n"
        
        # Assemble the complete prompt
        prompt_parts = [
            self.optimization_instructions,
            content_section,
            context_section if context_section else "",
            self.enhancement_instructions,
            self.output_format
        ]
        
        return "\\n".join(filter(None, prompt_parts))
    
    def _build_fast_prompt(self, raw_text: str, preprocessed: Dict) -> str:
        """
        Build a simplified prompt for faster processing.
        
        Args:
            raw_text: Original input text
            preprocessed: Preprocessed content data
            
        Returns:
            Optimized prompt string for fast processing
        """
        # Simplified fast prompt
        content_section = f"Content to Optimize:\\n{raw_text[:1000]}..."  # Limit content length
        
        prompt = f"""
Enhance this content with E-E-A-T principles. Add:
1. One citation (source/URL)
2. One expert quote
3. One statistic
Keep under 300 words, professional tone.

{content_section}

Enhanced content:
"""
        return prompt
    
    def optimize_content(self, raw_text: str, fast_mode: bool = True) -> str:
        """
        Optimize content using the GEO framework.
        
        Args:
            raw_text: Raw input text to optimize
            fast_mode: Use fast processing mode for better performance
            
        Returns:
            Optimized content with citations, quotes, and statistics
        """
        start_time = time.time()
        
        try:
            # 1. Classify input type
            t0 = time.time()
            content_type = self.content_processor.classify_input_type(raw_text)
            self.logger.debug(f"[Timing] classify_input_type: {time.time() - t0:.3f}s")
            
            # 2. Handle queries differently (fast path)
            if content_type == "query":
                return self._optimize_query(raw_text)
            
            # 3. Preprocess content for articles/blurbs with fast mode
            t1 = time.time()
            try:
                preprocessed = self.content_processor.preprocess_content(raw_text, fast_mode=fast_mode)
            except Exception as e:
                self.logger.error(f"Preprocessing failed: {e}", exc_info=True)
                return "Sorry, preprocessing failed. Please check your input or contact support."
            self.logger.debug(f"[Timing] preprocess_content: {time.time() - t1:.3f}s")
            
            # 4. Build optimization prompt (simplified for speed)
            t2 = time.time()
            prompt = self._build_fast_prompt(raw_text, preprocessed) if fast_mode else self.build_optimization_prompt(raw_text, preprocessed)
            self.logger.debug(f"[Timing] build_prompt: {time.time() - t2:.3f}s")
            
            # 5. Generate optimized content
            t3 = time.time()
            try:
                optimized_content = self.llm.invoke(prompt)
                self.logger.debug(f"[Timing] LLM invoke: {time.time() - t3:.3f}s")
                self.logger.info(f"[Timing] Total optimization: {time.time() - start_time:.3f}s")
                return optimized_content
            except Exception as e:
                self.logger.error(f"LLM invocation failed: {e}", exc_info=True)
                return "Sorry, we encountered an error while optimizing this content. Please try again later."
                
        except Exception as e:
            self.logger.error(f"Unexpected error in optimize: {e}", exc_info=True)
            return "An unexpected error occurred. Please contact support."
    
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
