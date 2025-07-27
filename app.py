import streamlit as st
from core.utils import is_url, fetch_text_from_url, classify_input_type
from core.gemini_chain import optimize, simulate_ai_response, generate_answer_for_query

st.set_page_config(page_title="Citation & Quotation Optimizer - PoC", layout="wide")

st.title("📚 AI-Driven Citation & Quotation Optimizer")
st.markdown(
    "Enhance your content with expert quotes, citations, and statistics for better generative engine visibility."
)


# --- Input Section ---
input_type = st.radio(
    "Select input type:",
    ["URL", "Raw Text"],
    help="Choose 'URL' to extract content from a web page, or 'Raw Text' to paste your own article."
)
user_input = st.text_area(
    "Enter your article URL or content:",
    value=st.session_state.get("current_input", ""),
    help="Paste a valid article URL or your own text here."
)
st.session_state.current_input = user_input
st.session_state.input_type = input_type


# Initialize session state
if "optimized" not in st.session_state:
    st.session_state.optimized = False
if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if "optimized_text" not in st.session_state:
    st.session_state.optimized_text = ""


# --- Processing Section ---
if st.button("Optimize"):
    if not st.session_state.current_input.strip():
        st.error("Please enter a valid URL or text.")
        st.stop()

    import time
    start_time = time.time()
    with st.spinner("Extracting content..."):
        try:
            if input_type == "URL":
                st.session_state.article_text = fetch_text_from_url(st.session_state.current_input)
            else:
                st.session_state.article_text = st.session_state.current_input.strip()
        except Exception as e:
            st.error(f"Error extracting from URL: {e}")
            st.stop()

    with st.spinner("Optimizing content ..."):
        try:
            st.session_state.optimized_text = optimize(st.session_state.article_text)
            st.session_state.optimized = True
            st.session_state.processing_time = time.time() - start_time
        except Exception as e:
            st.error(f"Error during optimization: {e}")
            st.stop()

# --- Output Section ---
if st.session_state.optimized:
    st.success("Optimization complete!")
    st.info(f"⏱️ Processing time: {st.session_state.get('processing_time', 0):.2f} seconds")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Content")
        st.text_area(
            "Original",
            st.session_state.article_text,
            height=800,
            key="original_text_area"
        )
    with col2:
        st.subheader("Optimized Content")
        st.text_area(
            "Optimized",
            st.session_state.optimized_text,
            height=800,
            key="optimized_text_area"
        )
    if st.button("Clear"):
        st.session_state.optimized = False
        st.session_state.article_text = ""
        st.session_state.optimized_text = ""
        st.session_state.current_input = ""
        st.session_state.processing_time = 0



# --- Simulated AI Response (remains commented for now)
# if st.session_state.article_text and st.session_state.optimized_text:
#     st.subheader("Simulated AI Responses")
#     user_query = st.text_input("Try a user query (e.g., 'What are current climate stats?')", "")
#     if user_query and st.button("Run Simulation"):
#         with st.spinner("Simulating AI response..."):
#             original_resp = simulate_ai_response(st.session_state.article_text, user_query)
#             optimized_resp = simulate_ai_response(st.session_state.optimized_text, user_query)
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("### Original Response")
#             st.write(original_resp)
#         with col2:
#             st.markdown("### Optimized Response")
#             st.write(optimized_resp)
