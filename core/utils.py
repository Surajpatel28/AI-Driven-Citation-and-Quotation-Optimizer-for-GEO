import re, nltk, spacy
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Download NLTK data if not already present
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.ErrorMessage:
    nltk.download(
        "punkt")  # Changed from punkt_tab to punkt, as punkt_tab is less common and punkt is usually sufficient.


# Load spaCy model with only NER enabled for speed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini is PaLM 2-like, use closest available
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")  # closest match


def estimate_token_count(text: str) -> int:
    return len(tokenizer.encode(text, truncation=False))


def get_max_output_tokens(text: str, content_type: str = "article") -> int:
    input_tokens = estimate_token_count(text)

    # Use adaptive multipliers per content type
    if content_type == "query":
        return min(512, 256 + input_tokens)  # Always short
    elif content_type == "blurb":
        return min(int(input_tokens * 1.25), 2048)
    elif content_type == "article":
        return min(int(input_tokens * 1.5), 8192)
    else:
        return min(int(input_tokens * 1.2), 1024)


def preprocess_content(text: str) -> dict:
    import time
    # 1. Advanced Cleaning: Remove boilerplate, deduplicate, filter non-English
    t0 = time.time()
    clean = re.sub(r'\s+', ' ', text).strip()
    boilerplate_patterns = [
        r'copyright \d{4}', r'all rights reserved', r'cookie policy', r'privacy policy',
        r'\bsubscribe\b', r'\blogin\b', r'\bsign up\b', r'\bterms of service\b'
    ]
    for pat in boilerplate_patterns:
        clean = re.sub(pat, '', clean, flags=re.IGNORECASE)
    print(f"[Timing] Cleaning: {time.time() - t0:.3f}s")

    if not clean:
        return {
            "clean_text": "",
            "sentences": [],
            "entities": [],
            "stats": [],
            "top_sentences": [],
            "weak_sentences": []
        }

    # 2. Sentence splitting, deduplication, and filtering
    t1 = time.time()
    try:
        sentences = nltk.sent_tokenize(clean)
    except Exception as e:
        print(f"NLTK sentence split failed: {e}")
        sentences = [clean]
    seen = set()
    filtered = []
    for s in sentences:
        s_clean = s.strip().lower()
        if len(s_clean) < 8 or s_clean in seen:
            continue
        seen.add(s_clean)
        filtered.append(s.strip())
    sentences = filtered
    print(f"[Timing] Sentence split/dedupe: {time.time() - t1:.3f}s")

    # 3. Entity extraction (expand to orgs, locations, dates)
    t2 = time.time()
    try:
        doc = nlp(clean)
        ents = [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        print(f"spaCy NER failed: {e}")
        ents = []
    print(f"[Timing] spaCy NER: {time.time() - t2:.3f}s")

    # 4. Stat extraction (expand patterns)
    t3 = time.time()
    stat_patterns = [
        r'\d+(?:\.\d+)?\s*(?:%|percent|million|billion|trillion|years?|people|cases|deaths|dollars|euros|pounds|students|citizens|votes|km|miles|liters|tons|hectares|acres)'
    ]
    stats = []
    for pat in stat_patterns:
        stats += re.findall(pat, clean, flags=re.IGNORECASE)
    print(f"[Timing] Stat extraction: {time.time() - t3:.3f}s")

    # 5. Weak claim detection (expanded)
    t4 = time.time()
    weak_patterns = [
        r"\bstudies show\b",
        r"\bsome people say\b",
        r"\bmany believe\b",
        r"\boften\b",
        r"\balways\b",
        r"\bnever\b",
        r"\bexperts claim\b",
        r"\bis the best\b",
        r"\bnumber one\b",
        r"\baccording to (?!\w+ \([\d]{4}\))",
        r"\b\d+%.*(prefer|agree|support|believe)\b",
        r"it is believed that",
        r"researchers suggest",
        r"recent studies",
        r"it is said that"
    ]
    weak_sentences = []
    for sent in sentences:
        for pattern in weak_patterns:
            if re.search(pattern, sent, flags=re.IGNORECASE):
                weak_sentences.append(sent)
                break
    print(f"[Timing] Weak claim detection: {time.time() - t4:.3f}s")

    # 6. Embedding & top sentence selection (centrality-based, batched)
    t5 = time.time()
    try:
        import numpy as np
        batch_size = 16
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            try:
                batch_emb = model.encode(batch)
                embeddings.extend(batch_emb)
            except Exception as e:
                print(f"Embedding batch failed: {e}")
        embeddings = np.array(embeddings)
        if embeddings.size == 0:
            top_sentences = []
        else:
            sim_matrix = np.inner(embeddings, embeddings)
            avg_sim = sim_matrix.mean(axis=1)
            top_idx = np.argsort(avg_sim)[-3:]
            top_sentences = [sentences[i] for i in sorted(top_idx)]
    except Exception as e:
        print(f"Embedding or top sentence selection failed: {e}")
        top_sentences = []
    print(f"[Timing] Embedding & top sentence: {time.time() - t5:.3f}s")

    return {
        "clean_text": clean,
        "sentences": sentences,
        "entities": ents,
        "stats": stats,
        "top_sentences": top_sentences,
        "weak_sentences": weak_sentences
    }

def classify_input_type(text: str) -> str:
    text = text.strip()
    word_count = len(text.split())

    # Basic query indicators
    query_keywords = [
        "what", "why", "how", "when", "where", "who", "which", "whom", "whose",
        "is", "are", "can", "do", "does", "should", "tell me", "explain", "give me"
    ]

    # Use regex to detect query phrases at start
    query_pattern = re.compile(r"^(?:" + "|".join(re.escape(q) for q in query_keywords) + r")\b", re.IGNORECASE)

    if text.endswith("?") or query_pattern.match(text):
        return "query"
    elif word_count < 40:
        return "blurb"
    else:
        return "article"


def is_url(text: str) -> bool:
    url_pattern = re.compile(
        r'^(https?://)?(www\.)?([A-Za-z_0-9-]+)+(\.[a-z]{2,})+(/[^\s]*)?$'
    )
    return bool(url_pattern.match(text.strip()))


def fetch_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n\n".join([p.get_text() for p in paragraphs])
        return text.strip()[:5000]  # limit for prompt
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed for URL {url}: {e}")
        return ""  # Return empty string on request failure
    except Exception as e:
        print(f"Failed to extract content from URL {url}: {e}")
        return ""  # Return empty string on other extraction failures
