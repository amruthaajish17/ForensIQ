# rag.py
import wikipediaapi
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# ── Embedding model ──────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # tiny + fast, runs locally

FORENSIC_TOPICS = [
    "Forensic science",
    "Blood spatter analysis",
    "Fingerprint evidence",
    "Forensic pathology",
    "DNA profiling",
    "Digital forensics",
    "Ballistics",
    "Trace evidence",
    "Crime scene investigation",
    "Forensic toxicology",
    "Chain of custody",
    "Locard's exchange principle",
    "Autopsy",
    "Forensic entomology",
    "questioned document examination",
]


# ── Wikipedia Corpus Loader ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="📚 Loading forensic knowledge from Wikipedia...")
def load_forensic_corpus() -> list[str]:
    """
    Pulls real forensic science content from Wikipedia.
    Cached so it only runs once per session.
    """
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="CrimeSceneAI/1.0 (capstone project)"
    )

    chunks = []
    for topic in FORENSIC_TOPICS:
        page = wiki.page(topic)
        if page.exists():
            paragraphs = [
                p.strip()
                for p in page.text.split("\n")
                if len(p.strip()) > 120  # skip stub lines
            ]
            chunks.extend(paragraphs[:15])  # top 15 paragraphs per topic
            print(f"✅ Loaded: {topic} → {len(paragraphs[:15])} chunks")
        else:
            print(f"⚠️  Skipped (not found): {topic}")

    print(f"\n📦 Total corpus size: {len(chunks)} chunks")
    return chunks


# ── FAISS Index Builder ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⚡ Building FAISS vector index...")
def build_faiss_index(_corpus: list[str]):
    """
    Embed corpus with sentence-transformers and index with FAISS.
    Underscore prefix on _corpus prevents Streamlit from hashing the list.
    """
    embeddings = embedder.encode(_corpus, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product = cosine after normalize
    index.add(embeddings)

    return index


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, index, corpus: list[str], top_k: int = 5) -> list[str]:
    """Retrieve top_k most relevant forensic passages for a query."""
    query_vec = embedder.encode([query], show_progress_bar=False).astype("float32")
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k)
    return [corpus[i] for i in indices[0] if i < len(corpus)]