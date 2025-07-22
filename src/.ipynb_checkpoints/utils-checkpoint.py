import re
import requests
from bs4 import BeautifulSoup

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