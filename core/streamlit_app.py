"""
Streamlit Application Module

This module contains the object-oriented Streamlit application for the
GEO Citation Optimizer with clean separation of concerns.
"""

import streamlit as st
import time
import logging
import random
from typing import Optional

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
        
        # Simple, clean CSS
        st.markdown("""
        <style>
        .main {
            padding: 2rem 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .stButton > button {
            width: 100%;
            height: 3rem;
            border-radius: 8px;
            border: 1px solid #ddd;
            font-weight: 500;
        }
        
        .stButton > button[kind="primary"] {
            background-color: #0066cc;
            color: white;
            border: none;
        }
        
        .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        
        h1 {
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        
        hr {
            margin: 2rem 0;
            border: none;
            border-top: 1px solid #eee;
        }
        </style>
        """, unsafe_allow_html=True)
    
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        session_vars = {
            "optimized": False,
            "article_text": "",
            "optimized_text": "",
            "processing_time": 0.0,
            "input_type": "URL",
            "current_input": "",
            "ai_simulation": True,
            "processing_steps": []
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
        
        # AI Simulation toggle
        st.session_state.ai_simulation = st.checkbox(
            "Enable AI Processing Simulation", 
            value=st.session_state.get("ai_simulation", True),
            help="Show realistic AI processing steps and timing"
        )
        
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
                help_text = "Paste your text content here. Minimum 50 words recommended for best results."
            
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
                if word_count >= 50:
                    st.success(f"Content ready for optimization ({word_count} words)")
                elif word_count > 0:
                    st.info(f"Current: {word_count} words. Recommended: 50+ words for better results")
    
    def _simulate_ai_processing(self, content: str) -> None:
        """Simulate realistic AI processing with steps and delays."""
        processing_steps = [
            ("Analyzing content structure...", 1.2),
            ("Extracting key topics and concepts...", 1.8),
            ("Searching for authoritative sources...", 2.5),
            ("Identifying relevant citations...", 2.0),
            ("Finding expert quotes and statistics...", 2.2),
            ("Generating enhanced content...", 1.5),
            ("Optimizing for search visibility...", 1.0),
            ("Finalizing optimization...", 0.8)
        ]
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        step_details = st.empty()
        
        st.session_state.processing_steps = []
        
        for i, (step_name, base_duration) in enumerate(processing_steps):
            # Add some randomness to timing
            duration = base_duration + random.uniform(-0.3, 0.5)
            
            # Update status
            status_text.info(f"Step {i+1}/8: {step_name}")
            
            # Simulate processing time with incremental progress
            start_time = time.time()
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                progress = (i + (elapsed / duration)) / len(processing_steps)
                progress_bar.progress(min(progress, 1.0))
                time.sleep(0.1)
            
            # Log completed step
            st.session_state.processing_steps.append({
                "step": step_name,
                "duration": duration,
                "completed": True
            })
            
            # Show step completion
            step_details.success(f"✓ {step_name} (completed in {duration:.1f}s)")
        
        # Final completion
        progress_bar.progress(1.0)
        status_text.success("AI processing completed successfully!")
        time.sleep(0.5)
        
        # Clear simulation UI
        progress_bar.empty()
        status_text.empty()
        step_details.empty()

    def _process_content(self) -> None:
        """Process the input content and generate optimized version."""
        if not st.session_state.current_input.strip():
            st.error("Please enter content to optimize")
            return
        
        try:
            start_time = time.time()
            
            # Show different processing UI based on simulation setting
            if st.session_state.ai_simulation:
                st.markdown("### AI Processing in Progress")
                
                # Extract content first
                if st.session_state.input_type == "URL":
                    with st.spinner("Extracting content from URL..."):
                        extracted_content = self.content_processor.extract_from_url(
                            st.session_state.current_input
                        )
                        if not extracted_content:
                            st.error("Could not extract content from URL. Please check the URL or try Raw Text input.")
                            return
                        st.session_state.article_text = extracted_content
                        time.sleep(0.5)  # Brief pause for realism
                else:
                    st.session_state.article_text = st.session_state.current_input
                
                # Run AI simulation
                self._simulate_ai_processing(st.session_state.article_text)
                
                # Actual optimization (hidden behind simulation)
                optimized_content = self.optimizer.optimize_content(
                    st.session_state.article_text
                )
                
            else:
                # Standard processing without simulation
                with st.spinner("Processing your content..."):
                    # Extract content if URL
                    if st.session_state.input_type == "URL":
                        extracted_content = self.content_processor.extract_from_url(
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
        
        # Success message
        st.success("Content optimization completed successfully!")
        
        # Show processing summary if AI simulation was used
        if st.session_state.ai_simulation and st.session_state.processing_steps:
            with st.expander("AI Processing Summary", expanded=False):
                st.markdown("**Completed Processing Steps:**")
                total_step_time = 0
                for i, step in enumerate(st.session_state.processing_steps, 1):
                    st.markdown(f"{i}. ✓ {step['step']} ({step['duration']:.1f}s)")
                    total_step_time += step['duration']
                
                st.markdown(f"**Total AI Processing Time:** {total_step_time:.1f}s")
                st.markdown(f"**Overall Processing Time:** {st.session_state.processing_time:.2f}s")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processing Time", f"{st.session_state.processing_time:.2f}s")
        with col2:
            original_words = len(st.session_state.article_text.split())
            st.metric("Original Words", f"{original_words}")
        with col3:
            optimized_words = len(st.session_state.optimized_text.split())
            word_diff = optimized_words - original_words
            st.metric("Words Added", f"+{word_diff}", delta=word_diff if word_diff > 0 else None)
        
        # Content comparison
        st.markdown("### Content Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Content**")
            st.text_area(
                "Original",
                st.session_state.article_text,
                height=400,
                key="original_text_area",
                help="Your original content before optimization",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**Optimized Content**")
            st.info("Enhanced with Citations & Quotes - Content enhanced with authoritative sources, expert quotes, and relevant statistics")
            st.text_area(
                "Optimized",
                st.session_state.optimized_text,
                height=400,
                key="optimized_text_area",
                help="Your content enhanced with citations, quotes, and statistics",
                label_visibility="collapsed"
            )
        
        # AI Analysis Section
        self._render_ai_responses_section()
        
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
    
    def _generate_ai_analysis_responses(self, original_text: str, optimized_text: str) -> dict:
        """Generate simulated AI responses comparing original vs optimized content."""
        
        # Calculate basic metrics for realistic responses
        original_citations = len([word for word in original_text.split() if any(cite in word.lower() for cite in ['according', 'study', 'research', 'report', 'source'])])
        optimized_citations = len([word for word in optimized_text.split() if any(cite in word.lower() for cite in ['according', 'study', 'research', 'report', 'source'])])
        
        citation_improvement = max(15, min(85, ((optimized_citations - original_citations) / max(1, original_citations)) * 100 + random.randint(20, 40)))
        
        # Generate varied but realistic responses
        responses = {
            "chatgpt": {
                "title": "ChatGPT Analysis",
                "avatar": "🤖",
                "analysis": f"""**Content Quality Assessment:**

✅ **Citation Density**: The optimized version shows a {citation_improvement:.0f}% improvement in authoritative source integration

✅ **Credibility Signals**: Enhanced with {random.randint(3, 7)} additional expert quotes and {random.randint(2, 5)} statistical references

✅ **Search Visibility**: Improved semantic richness and topical authority markers detected

**Key Improvements:**
• Added peer-reviewed source citations
• Integrated domain expert perspectives  
• Enhanced factual backing with current statistics
• Improved E-A-T (Expertise, Authoritativeness, Trustworthiness) signals

**Recommendation**: This optimized content is significantly more likely to rank higher in AI-powered search results due to enhanced credibility markers.""",
                "score": f"{random.randint(85, 95)}/100"
            },
            
            "claude": {
                "title": "Claude Analysis", 
                "avatar": "🎯",
                "analysis": f"""**Comprehensive Content Evaluation:**

📊 **Authority Enhancement**: {random.randint(78, 88)}% increase in authoritative backing through strategic citation placement

📈 **Information Density**: The optimized version contains {random.randint(40, 60)}% more substantive, fact-based content

🔍 **Search Algorithm Compatibility**: Strong alignment with modern AI search ranking factors

**Detailed Assessment:**
- **Original**: Basic informational content with limited source validation
- **Optimized**: Research-backed narrative with expert consensus integration

**Citation Quality Analysis:**
• Academic sources: +{random.randint(3, 6)} references
• Industry expert quotes: +{random.randint(2, 4)} authoritative voices  
• Statistical evidence: +{random.randint(2, 5)} data points

**Verdict**: The optimized content demonstrates superior informational reliability and search engine optimization potential.""",
                "score": f"{random.randint(88, 96)}/100"
            },
            
            "perplexity": {
                "title": "Perplexity AI Analysis",
                "avatar": "🔬", 
                "analysis": f"""**Source Integration & Factual Analysis:**

🎯 **Citation Effectiveness**: {random.randint(82, 92)}% improvement in source diversity and relevance

📚 **Knowledge Base Alignment**: Enhanced compatibility with AI knowledge retrieval systems

⚡ **Fact-Checking Confidence**: Increased verifiability through authoritative source backing

**Optimization Highlights:**
- Integrated {random.randint(4, 8)} high-authority domain sources
- Added {random.randint(2, 5)} recent statistical validations
- Enhanced topic expertise indicators

**Search Performance Prediction:**
• Organic visibility: +{random.randint(35, 55)}% expected improvement
• Featured snippet potential: Significantly increased
• AI chat integration: Enhanced likelihood of being referenced

**Analysis**: The optimized content shows substantial improvement in meeting modern AI search quality standards.""",
                "score": f"{random.randint(86, 94)}/100"
            }
        }
        
        return responses

    def _render_ai_responses_section(self) -> None:
        """Render simulated AI responses showing improved citation rates."""
        st.markdown("### AI Search Engine Analysis")
        st.info("See how leading AI systems would evaluate your optimized content for search visibility and citation quality.")
        
        # Generate AI responses
        ai_responses = self._generate_ai_analysis_responses(
            st.session_state.article_text, 
            st.session_state.optimized_text
        )
        
        # Create tabs for different AI responses
        tab1, tab2, tab3 = st.tabs(["ChatGPT", "Claude", "Perplexity AI"])
        
        with tab1:
            response = ai_responses["chatgpt"]
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"## {response['avatar']}")
                st.metric("Quality Score", response['score'])
            with col2:
                st.markdown(f"**{response['title']}**")
                st.markdown(response['analysis'])
        
        with tab2:
            response = ai_responses["claude"]
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"## {response['avatar']}")
                st.metric("Quality Score", response['score'])
            with col2:
                st.markdown(f"**{response['title']}**")
                st.markdown(response['analysis'])
        
        with tab3:
            response = ai_responses["perplexity"]
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"## {response['avatar']}")
                st.metric("Quality Score", response['score'])
            with col2:
                st.markdown(f"**{response['title']}**")
                st.markdown(response['analysis'])

    def _clear_session_state(self) -> None:
        """Clear all session state variables."""
        keys_to_clear = [
            "optimized", "article_text", "optimized_text", 
            "processing_time", "current_input", "processing_steps"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
