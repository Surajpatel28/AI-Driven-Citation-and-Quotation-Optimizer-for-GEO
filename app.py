import streamlit as st
from src.utils import is_url, fetch_text_from_url
from src.gemini_chain import optimize, simulate_ai_response

st.set_page_config(page_title="Citation & Quotation Optimizer - PoC", layout="wide")

st.title("📚 AI-Driven Citation & Quotation Optimizer")
st.markdown(
    "Enhance your content with expert quotes, citations, and statistics for better generative engine visibility."
)

# --- Input
input_type = st.radio("Select input type:", ["URL", "Raw Text"])
user_input = st.text_area("Enter your article URL or content:")

# Initialize session state
if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if "optimized_text" not in st.session_state:
    st.session_state.optimized_text = ""

if st.button("Optimize"):
    if not user_input.strip():
        st.error("Please enter a valid URL or text.")
    else:
        with st.spinner("Extracting content..."):
            if input_type == "URL":
                try:
                    st.session_state.article_text = fetch_text_from_url(user_input)
                except Exception as e:
                    st.error(f"Error extracting from URL: {e}")
                    st.stop()
            else:
                st.session_state.article_text = user_input.strip()

        with st.spinner("Optimizing content ..."):
            try:
                st.session_state.optimized_text = optimize(st.session_state.article_text)
            except Exception as e:
                st.error(f"Error during optimization: {e}")
                st.stop()

        # --- Output
        st.success("Optimization complete!")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Content")
            st.write(st.session_state.article_text)
        with col2:
            st.subheader("Optimized Content")
            st.write(st.session_state.optimized_text)

# --- Simulated AI Response
if st.session_state.article_text and st.session_state.optimized_text:
    st.subheader("Simulated AI Responses")
    user_query = st.text_input("Try a user query (e.g., 'What are current climate stats?')", "")
    if user_query and st.button("Run Simulation"):
        with st.spinner("Simulating AI response..."):
            original_resp = simulate_ai_response(st.session_state.article_text, user_query)
            optimized_resp = simulate_ai_response(st.session_state.optimized_text, user_query)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Original Response")
            st.write(original_resp)
        with col2:
            st.markdown("### Optimized Response")
            st.write(optimized_resp)
