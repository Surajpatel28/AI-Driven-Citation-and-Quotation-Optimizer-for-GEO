import os 
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load Environment variables
load_dotenv()

#Initialize Gemini LLM
llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["content"],
    template="""
    You are an expert AI writing assistant trained to improve the visibility of web content in generative engines like ChatGPT, Gemini, and Claude.

Your task is to enhance the following article content by:
1. Adding **authoritative citations** from well-known organizations (e.g., WHO, UN, McKinsey, Nature, NYTimes, etc.)
2. Inserting **expert quotations** where relevant (real or convincingly generated with proper attribution)
3. Including **relevant statistics** or data-backed facts to support key points
4. Rewriting weak or unsupported claims to improve credibility, trust, and alignment with **E-E-A-T** standards (Experience, Expertise, Authoritativeness, Trustworthiness)

Guidelines:
- Keep the tone and structure of the original text
- Don't invent wild facts — be plausible and credible
- Ensure any additions look natural, not forced
- Try to improve SEO keywords if appropriate, but subtly

Content to optimize:
--------------------
{content}
--------------------

Now return the fully optimized version of the article.
If there is no content, do nothing.
    
.
    """
)

# create a Langchain LLMChain
gemini_chain = prompt_template | llm 

# create a function for using the chain
def optimize(article_text: str) -> str:
    return gemini_chain.invoke({"content": article_text})

def simulate_ai_response(content: str, user_query: str) -> str:
    prompt = f"""
    You are an AI assistant answering the following user query:

    User Query: "{user_query}"

    Use the content below to generate your response.
    Content:
    {content}

    Only use facts from the provided content. Be helpful and cite relevant parts clearly.
    """
    return gemini_chain.invoke(prompt) 