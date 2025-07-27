import os
import time
import logging
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'error.log')
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s %(message)s'
)

try:
    from utils import classify_input_type, preprocess_content
except ImportError:
    from core.utils import classify_input_type, preprocess_content
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load Environment variables
load_dotenv()

 # Initialize Gemini LLM once (singleton)
llm = GoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.4)

# dynamic prompt building
def build_prompt(raw_text, preprocessed):
    # Modular prompt sections
    instructions = (
        "IMPORTANT: Every output MUST include at least one authoritative citation (URL, source, or reference), one expert quote, and one relevant statistic. Outputs missing any of these will be considered incomplete.\n\n"
        "You are an expert AI content editor trained in Generative Engine Optimization (GEO), specializing in enhancing content for large language models like Google Gemini, ChatGPT, Claude, and Perplexity.\n\n"
        "Your goal is to optimize the following content by automatically injecting authoritative citations, expert quotations, relevant statistics, and institutional references. As shown in the 2025 GEO framework updates, this can improve visibility in AI-driven search by up to 40%.\n\n"
        "The optimization must follow E-E-A-T principles (Experience, Expertise, Authoritativeness, Trustworthiness) and should preserve the original meaning and tone while improving credibility, fluency, uniqueness, and engagement.\n"
    )

    # Few-shot example (edit as needed for your domain)
    few_shot_example = (
        "Example Output:\n"
        "Global warming is the long-term rise in Earth's average surface temperature, primarily due to human activities (NASA, 2023). According to Dr. Jane Smith, climate scientist at the UN, \"The evidence for rapid climate change is compelling.\" In 2022, global CO2 levels reached 419 ppm, the highest in 800,000 years (Source: https://climate.nasa.gov/vital-signs/carbon-dioxide/).\n"
    )
     # Content section
    sentences = " ".join(preprocessed["sentences"])
    content_section = f"Content to Optimize:\n{sentences}\n"

    # Context section (only include if available)
    context_parts = []
    if preprocessed.get("top_sentences"):
        top_clips = "\n".join(preprocessed["top_sentences"])
        if top_clips.strip():
            context_parts.append(f"Top Sentences to Emphasize:\n{top_clips}")
    if preprocessed.get("weak_sentences"):
        weak_claims = " ".join(preprocessed["weak_sentences"])
        if weak_claims.strip():
            context_parts.append(f"Flagged Weak Claims (for strengthening or citation):\n{weak_claims}")
    if preprocessed.get("stats"):
        stats = ", ".join(preprocessed["stats"])
        if stats.strip():
            context_parts.append(f"Relevant Statistics Found:\n{stats}")
    context_section = "\n\n".join(context_parts)
    if context_section:
        context_section = f"Context to Guide Enhancement:\n{context_section}\n"

    # Enhancement instructions (modular, easy to update)
    enhancement_instructions = """
    Enhancement Instructions:
    1. Begin with a concise definition or summary.
    2. Integrate a direct expert quote with attribution.
    3. Include a recent, relevant statistic with source.
    4. Add at least one authoritative citation (URL, journal, or institution).
    5. Structure the output clearly (paragraphs, bullet points if needed).
    6. Keep the output under 500 words.
    7. Do NOT include explanations or commentary—only the optimized content.

    Checklist (ensure all are present):
    - [ ] At least one citation
    - [ ] At least one expert quote
    - [ ] At least one statistic
    - [ ] Clear structure
    """
    # Output format and constraints
    output_format = (
        "Output Format:\n"
        "Return only the final optimized content, rewritten with the above goals in mind. Do not include explanation or commentary.\n"
        "Keep the output under 500 words.\n"
    )

    # Assemble the prompt
    prompt = "\n".join([
        instructions,
        few_shot_example,
        content_section,
        context_section if context_section else "",
        enhancement_instructions,
        output_format
    ])
    return prompt

# create a function for using the chain
def optimize(raw_text: str) -> str:
    t0 = time.time()
    try:
        content_type = classify_input_type(raw_text)
        print(f"[Timing] classify_input_type: {time.time() - t0:.3f}s")
        if content_type == "query":
            return generate_answer_for_query(raw_text)

        # For article or blurb
        t1 = time.time()
        try:
            preprocessed = preprocess_content(raw_text)
        except Exception as e:
            logging.error(f"Preprocessing failed: {e}", exc_info=True)
            return "Sorry, preprocessing failed. Please check your input or contact support."
        print(f"[Timing] preprocess_content: {time.time() - t1:.3f}s")

        t2 = time.time()
        prompt = build_prompt(raw_text, preprocessed)
        print(f"[Timing] build_prompt: {time.time() - t2:.3f}s")

        t3 = time.time()
        try:
            optimized_content = llm.invoke(prompt)
            print(f"[Timing] LLM invoke: {time.time() - t3:.3f}s")
            return optimized_content
        except Exception as e:
            logging.error(f"LLM invocation failed: {e}", exc_info=True)
            return "Sorry, we encountered an error while optimizing this content. Please try again later."
    except Exception as e:
        logging.error(f"Unexpected error in optimize: {e}", exc_info=True)
        return "An unexpected error occurred. Please contact support."

def generate_answer_for_query(query: str) -> str:
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""
        You're an expert content optimizer.
        Answer the following question in a way that is highly optimized for generative engines like ChatGPT:
            - Include citations (sources like UN, WHO, etc.)
            - Include relevant statistics and data
            - Add expert quotes if relevant
            - Write clearly and concisely
        Keep the total response length under 200 words but include at least one quote, one stat, and one citation.

        Question: {content}
    """
    )
    query_chain = prompt | llm
    t0 = time.time()
    try:
        optimized_query_response = query_chain.invoke({"content": query})
        print(f"[Timing] LLM Query invoke: {time.time() - t0:.3f}s")
        print(f"LLM Query Output: {optimized_query_response[:500]}...")  # Print first 500 chars for debugging
        return optimized_query_response
    except Exception as e:
        logging.error(f"LLM invocation failed for query: {e}", exc_info=True)
        return "Sorry, we encountered an error while optimizing this content for your query. Please try again later."

def simulate_ai_response(content: str, user_query: str) -> str:
    # Use singleton LLM
    prompt = f"""
    You are an AI assistant answering the following user query:

    User Query: "{user_query}"

    Use the content below to generate your response.
    Content:
    {content}

    Only use facts from the provided content. Be helpful and cite relevant parts clearly.
    """
    try:
        return llm.invoke(prompt)
    except Exception as e:
        logging.error(f"LLM invocation failed in simulate_ai_response: {e}", exc_info=True)
        return "Sorry, we encountered an error while generating the AI response. Please try again later."