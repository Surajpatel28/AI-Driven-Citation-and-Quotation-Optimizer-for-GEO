"""
Streamlit Application Module

This module contains the object-oriented Streamlit application for the
GEO Citation Optimizer with clean separation of concerns.
"""

import streamlit as st
import time
import logging
from typing import Optional, Dict

from core.content_processor import ContentProcessor
from core.geo_optimizer import GEOOptimizer


class StreamlitApp:
    """
    Object-oriented Streamlit application for GEO Citation Optimizer.
    
    This class manages the Streamlit interface and user interactions
    in a clean, maintainable way.
    """
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.logger = self._setup_logging()
        self._setup_page_config()
        self._initialize_session_state()
        self._initialize_components()
    
    def run(self) -> None:
        """Main application runner."""
        try:
            self.render_header()
            self.render_input_section()
            
            # Process content when button is clicked
            if st.button("Optimize Content", type="primary", use_container_width=True):
                self._process_content()
            
            # Render results if available
            self.render_output_section()
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            st.error(f"An error occurred: {e}")
    
    def _initialize_components(self) -> None:
        """Initialize cached components."""
        self.content_processor = self._get_content_processor()
        self.optimizer = self._get_geo_optimizer()
    
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
    
    def _setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="AI Citation & Quote Optimizer", 
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        session_vars = {
            "optimized": False,
            "article_text": "",
            "optimized_text": "",
            "processing_time": 0.0,
            "input_type": "URL",
            "current_input": ""
        }
        
        for var, default_value in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default_value
    
    @st.cache_resource
    def _get_content_processor(_self) -> ContentProcessor:
        """Get cached ContentProcessor instance."""
        return ContentProcessor()
    
    @st.cache_resource  
    def _get_geo_optimizer(_self) -> GEOOptimizer:
        """Get cached GEOOptimizer instance."""
        return GEOOptimizer()
    
    def render_header(self) -> None:
        """Render the application header."""
        st.title("AI Citation & Quote Optimizer")
        st.markdown('<p class="subtitle">Transform your content with authoritative citations, expert quotes, and relevant statistics for maximum AI search visibility</p>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_input_section(self) -> None:
        """Render the input section of the application."""
        st.subheader("Content Input")
        
        # Input type selection
        col1, col2 = st.columns([1, 3])
        with col1:
            st.session_state.input_type = st.radio(
                "Input Type:",
                ["URL", "Raw Text"],
                help="Choose 'URL' to extract content from a web page, or 'Raw Text' to paste your own article."
            )
        
        with col2:
            # Dynamic placeholder based on input type
            if st.session_state.input_type == "URL":
                placeholder = "Enter article URL (e.g., https://example.com/article)"
                help_text = "Paste a valid article URL here. We'll extract and analyze the content automatically."
            else:
                placeholder = "Paste your article content here..."
                help_text = "Paste your text content here for GEO optimization."
            
            st.session_state.current_input = st.text_area(
                f"Enter your {st.session_state.input_type.lower()}:",
                value=st.session_state.get("current_input", ""),
                placeholder=placeholder,
                help=help_text,
                height=200
            )
        
        # Input validation
        if st.session_state.current_input:
            if st.session_state.input_type == "URL":
                if st.session_state.current_input.startswith(('http://', 'https://')):
                    st.success("Valid URL format detected")
                else:
                    st.warning("Please enter a valid URL starting with http:// or https://")
            else:
                word_count = len(st.session_state.current_input.split())

    def _process_content(self) -> None:
        """Process the input content and generate optimized version."""
        if not st.session_state.current_input.strip():
            st.error("Please enter content to optimize")
            return
        
        try:
            start_time = time.time()
            
            # Standard processing
            with st.spinner("Processing your content..."):
                # Extract content if URL
                if st.session_state.input_type == "URL":
                    extracted_content = self.content_processor.fetch_text_from_url(
                        st.session_state.current_input
                    )
                    if not extracted_content:
                        st.error("Could not extract content from URL. Please check the URL or try Raw Text input.")
                        return
                    st.session_state.article_text = extracted_content
                else:
                    st.session_state.article_text = st.session_state.current_input
                
                # Optimize content
                optimized_content = self.optimizer.optimize_content(
                    st.session_state.article_text
                )
            
            # Update session state
            st.session_state.optimized_text = optimized_content
            st.session_state.processing_time = time.time() - start_time
            st.session_state.optimized = True
            
            st.rerun()
            
        except Exception as e:
            self.logger.error(f"Error processing content: {e}")
            st.error(f"Error processing content: {str(e)}")
    
    def render_output_section(self) -> None:
        """Render the output section with optimized content."""
        if not st.session_state.optimized:
            return
        
        st.markdown("---")
        st.subheader("Optimization Results")
        
        # Content comparison
        st.markdown("### Content Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Content**")
            with st.container():
                st.markdown("---")
                st.markdown(st.session_state.article_text)
        
        with col2:
            st.markdown("**Optimized Content**")
            with st.container():
                st.markdown("---")
                st.markdown(st.session_state.optimized_text, unsafe_allow_html=True)
        
        # Interactive Query Section
        self._render_interactive_query_section()
        
        # Action buttons
        st.markdown("### Actions")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Clear All", help="Clear all content and start over"):
                self._clear_session_state()
                st.rerun()
        with col3:
            st.download_button(
                label="Download Optimized Content",
                data=st.session_state.optimized_text,
                file_name="optimized_content.txt",
                mime="text/plain",
                help="Download your optimized content as a text file"
            )

    def _clear_session_state(self) -> None:
        """Clear all session state variables."""
        keys_to_clear = [
            "optimized", "article_text", "optimized_text", 
            "processing_time", "current_input"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    def _render_interactive_query_section(self) -> None:
        """Render interactive query section where users can ask questions about the content."""
        st.markdown("---")
        st.markdown("### Ask AI About Your Content")
        st.markdown("Ask any question about your content and see how AI responds using original vs optimized content.")
        
        # Query input
        user_query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the main benefits of this approach?",
            key="user_query"
        )
        
        if user_query and st.button("🚀 Get AI Response", type="primary"):
            with st.spinner("Getting responses from Gemini AI..."):
                try:
                    # Get responses for both contents
                    original_response = self._get_gemini_response(user_query, st.session_state.article_text)
                    optimized_response = self._get_gemini_response(user_query, st.session_state.optimized_text)
                    
                    # Display responses side by side
                    st.markdown("#### AI Responses Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Response from Original Content**")
                        st.markdown(original_response)
                    
                    with col2:
                        st.markdown("**Response from Optimized Content**")
                        st.markdown(optimized_response)
                        
                except Exception as e:
                    st.error(f"Error getting AI response: {str(e)}")
    
    def _get_gemini_response(self, query: str, content: str) -> str:
        """Get response from Gemini API for a given query and content."""
        try:
            # Create prompt for Gemini
            prompt = f"""
            You are an AI assistant answering the following user query based on the provided content.

            User Query: "{query}"

            Content to base your answer on:
            {content}

            Instructions:
            - Answer the query using only information from the provided content
            - Be concise and helpful
            - If the content doesn't contain relevant information, say so
            - Cite specific parts of the content when relevant
            - Keep response under 200 words

            Response:
            """
            
            # Use the existing optimizer's LLM
            response = self.optimizer.llm.invoke(prompt)
            return response.strip()
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
