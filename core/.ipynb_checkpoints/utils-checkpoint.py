import re
import requests
from bs4 import BeautifulSoup

def classify_input_type(text: str) -> str:
    text = text.strip()
    word_count = len(text.split())

    # Basic query indicators
    query_indicators = ["what", "why", "how", "when", "where", "who", "which", "whom", "whose","is", "are", "can", "do"]

    if word_count < 12 and text.endswith("?"):
        return "query"
    elif word_count < 40:
        return "blurb"
    else:
        return "article"

def is_url(text: str)->bool:
    url_pattern = re.compile(
        r'^(https?://)?(www\.)?([A-Za-z_0-9-]+)+(\.[a-z]{2,})+(/[^\s]*)?$'
    )
    return bool(url_pattern.match(text.strip()))

def fetch_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n\n".join([p.get_text() for p in paragraphs])
        return text.strip()[:5000]  # limit for prompt
    except Exception as e:
        return f"Failed to extract content: {e}"