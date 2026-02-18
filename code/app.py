# code/app.py
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import math

import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_docs import DOCS, DOCS_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
EMBED_FILE = BASE_DIR / "chunks_with_embeddings_all_local.jsonl"

# Bi-encoder for fast retrieval
BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"#"sentence-transformers/all-mpnet-base-v2"

# Cross-encoder for semantic similarity
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"# "cross-encoder/stsb-roberta-large"

TOP_K_CHUNKS = 5
CANDIDATE_CHUNKS = 50
TOP_K_DOCS = 5

# Stopwords for keyword extraction
STOPWORDS = {
    "the", "and", "or", "a", "an", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "it", "its", "as", "about", "into", "through", "over", "under",
    "such", "than", "then", "so", "if", "but", "not", "no", "yes",
    "can", "could", "should", "would", "may", "might",
    "we", "you", "they", "he", "she", "i", "our", "your", "their",
    "also", "there", "therefore", "thus",
    "using", "used", "use",
    "data", "information"
}

app = Flask(__name__)

# Global variables for models and index
bi_encoder = None
cross_encoder = None
records = None
matrix = None


def load_index(embed_file: Path):
    """Load embeddings and records from JSONL file."""
    records_list = []
    embeddings = []

    with embed_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            emb = np.array(rec["embedding"], dtype="float32")
            records_list.append(rec)
            embeddings.append(emb)

    if not embeddings:
        raise RuntimeError("No embeddings found in file.")

    matrix = np.vstack(embeddings)
    return records_list, matrix


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all embeddings."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)


def text_to_keywords(text: str) -> set:
    """Extract keywords from text."""
    tokens = re.split(r"\W+", text.lower())
    keywords = {t for t in tokens if t and len(t) > 2 and t not in STOPWORDS}
    return keywords


def clean_query(query: str) -> str:
    """Remove conversational fluff from queries."""
    # Remove common conversational phrases
    #PROMPT ENGINEERING
    fluff_patterns = [
        r'\bi\s+wanna\s+know\s+about\s+',
        r'\btell\s+me\s+about\s+',
        r'\bwhat\s+is\s+',
        r'\bwhat\s+are\s+',
        r'\bshow\s+me\s+',
        r'\bfind\s+me\s+',
        r'\bsearch\s+for\s+',
        r'\bcan\s+you\s+',
        r'\bplease\s+',
        r'\bexplain\s+',
        r'\bsuggest\s+me\s+articles\s+on\s+',
    ]
    
    cleaned = query.lower().strip()
    for pattern in fluff_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip() or query  # Return original if cleaning removes everything


def search(query: str):
    """Perform semantic search and return results."""
    if not query or not query.strip():
        return {"chunks": [], "documents": [], "error": "Query cannot be empty"}

    query = query.strip()
    # Clean conversational fluff from query
    cleaned_query = clean_query(query)
    
    # Extract keywords from query
    query_keywords = text_to_keywords(cleaned_query)

    # Step 1: Bi-encoder retrieval
    query_emb = bi_encoder.encode([cleaned_query])[0]
    sims = cosine_sim(query_emb, matrix)

    # Get candidate indices by cosine similarity
    num_candidates = min(CANDIDATE_CHUNKS, len(records))
    candidate_idx = np.argsort(-sims)[:num_candidates]

    # Build (query, text) pairs for cross-encoder
    candidate_pairs = []
    for idx in candidate_idx:
        text = records[idx]["text"]
        text_for_ce = text[:1200]
        candidate_pairs.append((cleaned_query, text_for_ce))

    # Step 2: Cross-encoder similarity
    # ----- STEP 2: cross-encoder similarity -----
    cross_scores = cross_encoder.predict(candidate_pairs)
    cross_scores = [1 / (1 + math.exp(-score)) for score in cross_scores]
    # Collect candidates with both scores + keyword overlap
    candidates = []
    for i, idx in enumerate(candidate_idx):
        rec = records[idx]
        chunk_text = rec["text"]
        chunk_keywords = text_to_keywords(chunk_text)
        overlap = query_keywords.intersection(chunk_keywords)
        overlap_count = len(overlap)

        candidates.append({
            "idx": idx,
            "record": rec,
            "bi_score": float(sims[idx]),
            "cross_score": float(cross_scores[i]),
            "overlap_count": overlap_count,
        })

    # Step 2.5: Keyword-overlap filtering
    filtered_candidates = [c for c in candidates if c["overlap_count"] > 0]

    if filtered_candidates:
        candidates_to_rank = filtered_candidates
    else:
        candidates_to_rank = candidates

    # Step 3: Sort by cross-encoder score
    candidates_to_rank.sort(key=lambda x: x["cross_score"], reverse=True)

    # Step 4: Keep only the best-scoring chunk per document
    # This prevents multiple chunks from the same doc from showing up.
    seen_docs = set()
    unique_chunks = []
    for cand in candidates_to_rank:
        doc_id = cand["record"]["doc_id"]
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        unique_chunks.append(cand)
        if len(unique_chunks) == TOP_K_CHUNKS:
            break

    top_chunks = unique_chunks

    # Format chunks for display
    formatted_chunks = []
    for rank, c in enumerate(top_chunks, start=1):
        rec = c["record"]
        chunk_data = {
            "rank": rank,
            "doc_id": rec["doc_id"],
            "chunk_id": rec["chunk_id"],
            "page": rec["page"],
            "text": rec["text"],
            "bi_score": c["bi_score"],
            "cross_score": c["cross_score"],
            "overlap_count": c["overlap_count"],
            "document_title": rec.get("document_title", "").strip() or "Untitled Document"  
   
        }
        # Add document title if available and not empty
        if "document_title" in rec and rec["document_title"] and rec["document_title"].strip():
            chunk_data["document_title"] = rec["document_title"].strip()
        formatted_chunks.append(chunk_data)

    # Step 5: Document-level recommendations
    doc_scores = defaultdict(list)
    for c in candidates_to_rank:
        doc_id = c["record"]["doc_id"]
        doc_scores[doc_id].append(c["cross_score"])

    doc_score_agg = {doc_id: max(scores) for doc_id, scores in doc_scores.items()}
    top_docs = sorted(doc_score_agg.items(), key=lambda x: x[1], reverse=True)[:TOP_K_DOCS]

    formatted_docs = []
    for rank, (doc_id, score) in enumerate(top_docs, start=1):
        formatted_docs.append({
            "rank": rank,
            "doc_id": doc_id,
            "similarity": float(score),
        })

    return {
        "chunks": formatted_chunks,
        "documents": formatted_docs,
    }


@app.route("/")
def index():
    """Render the main search page."""
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search_endpoint():
    """Handle search requests."""
    data = request.get_json()
    query = data.get("query", "")

    try:
        results = search(query)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/view/<doc_id>")
def view_document(doc_id):
    """Serve PDF document by doc_id."""
    # Find the document in DOCS config
    doc_config = None
    for doc in DOCS:
        if doc["doc_id"] == doc_id:
            doc_config = doc
            break
    
    if not doc_config:
        return jsonify({"error": "Document not found"}), 404
    
    pdf_path = doc_config.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        return jsonify({"error": "PDF file not found"}), 404
    
    return send_file(
        str(pdf_path),
        mimetype="application/pdf",
        as_attachment=False,
        download_name=Path(pdf_path).name
    )


def init_app():
    """Initialize models and load index."""
    global bi_encoder, cross_encoder, records, matrix

    print(f"Loading bi-encoder: {BI_ENCODER_MODEL}")
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)

    print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    print(f"Loading embeddings from {EMBED_FILE}")
    records, matrix = load_index(EMBED_FILE)
    print(f"Loaded {len(records)} chunks")


init_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

