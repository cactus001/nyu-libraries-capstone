# code/semantic_search_multi.py
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np            # pip install numpy
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
EMBED_FILE = BASE_DIR / "chunks_with_embeddings_all_local.jsonl"

# Bi-encoder for fast retrieval (same as used in embedding step)
BI_ENCODER_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Cross-encoder for semantic similarity (0–1-ish scores)
CROSS_ENCODER_MODEL = "cross-encoder/stsb-roberta-large"

TOP_K_CHUNKS = 5       # final chunks to show
CANDIDATE_CHUNKS = 50  # how many chunks to rerank with cross-encoder
TOP_K_DOCS = 5         # top documents to recommend

# Very simple English stopword list for keyword extraction
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
    "data", "information"  # you can remove these if they are important domain words
}


def load_index(embed_file: Path):
    records = []
    embeddings = []

    with embed_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            emb = np.array(rec["embedding"], dtype="float32")
            records.append(rec)
            embeddings.append(emb)

    if not embeddings:
        raise RuntimeError("No embeddings found in file.")

    matrix = np.vstack(embeddings)
    return records, matrix


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)


def text_to_keywords(text: str) -> set:
    """
    Simple keyword extractor:
    - lowercase
    - split on non-word chars
    - drop stopwords
    - drop very short tokens
    """
    tokens = re.split(r"\W+", text.lower())
    keywords = {t for t in tokens if t and len(t) > 2 and t not in STOPWORDS}
    return keywords


def main():
    print(f"Loading bi-encoder: {BI_ENCODER_MODEL}")
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)

    print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    print(f"Loading embeddings from {EMBED_FILE}")
    records, matrix = load_index(EMBED_FILE)
    print(f"Loaded {len(records)} chunks")

    while True:
        query = input("\nEnter your question (or press Enter to quit): ").strip()
        if not query:
            print("Goodbye!")
            break

        # Extract keywords from query once
        query_keywords = text_to_keywords(query)

        # ----- STEP 1: bi-encoder retrieval -----
        query_emb = bi_encoder.encode([query])[0]
        sims = cosine_sim(query_emb, matrix)

        # Get candidate indices by cosine similarity
        num_candidates = min(CANDIDATE_CHUNKS, len(records))
        candidate_idx = np.argsort(-sims)[:num_candidates]

        # Build (query, text) pairs for cross-encoder
        candidate_pairs = []
        for idx in candidate_idx:
            text = records[idx]["text"]
            # Optional: truncate long text for safety (char-level)
            text_for_ce = text[:1200]
            candidate_pairs.append((query, text_for_ce))

        # ----- STEP 2: cross-encoder similarity -----
        cross_scores = cross_encoder.predict(candidate_pairs)

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

        # ----- STEP 2.5: keyword-overlap filtering -----
        # Filter out chunks that share zero keywords with the query.
        filtered_candidates = [c for c in candidates if c["overlap_count"] > 0]

        # If everything got filtered out, fall back to the unfiltered list
        if filtered_candidates:
            candidates_to_rank = filtered_candidates
        else:
            print("⚠️ No keyword overlap found; falling back to all candidates.")
            candidates_to_rank = candidates

        # ----- STEP 3: sort by cross-encoder score -----
        candidates_to_rank.sort(key=lambda x: x["cross_score"], reverse=True)

        # ----- STEP 4: keep only best chunk per document -----
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

        print(f"\nTop {len(top_chunks)} chunks (by STS similarity + keyword filter):\n")
        for rank, c in enumerate(top_chunks, start=1):
            rec = c["record"]
            doc_id = rec["doc_id"]
            page = rec["page"]
            chunk_id = rec["chunk_id"]
            text = rec["text"]
            bi_score = c["bi_score"]
            cross_score = c["cross_score"]
            overlap_count = c["overlap_count"]

            snippet = text[:400].replace("\n", " ") + ("..." if len(text) > 400 else "")

            print(
                f"{rank}. doc_id={doc_id} | chunk_id={chunk_id} | page={page} | "
                f"bi_cosine={bi_score:.3f} | similarity={cross_score:.3f} | "
                f"keyword_overlap={overlap_count}"
            )
            print(f"   {snippet}\n")

        # ----- STEP 5: document-level recommendations -----
        # Aggregate similarity per doc_id over the ranked candidates
        doc_scores = defaultdict(list)
        for c in candidates_to_rank:
            doc_id = c["record"]["doc_id"]
            doc_scores[doc_id].append(c["cross_score"])

        # Use max similarity as the document relevance
        doc_score_agg = {doc_id: max(scores) for doc_id, scores in doc_scores.items()}

        top_docs = sorted(doc_score_agg.items(), key=lambda x: x[1], reverse=True)[:TOP_K_DOCS]

        print(f"Top {len(top_docs)} documents for this query (by STS similarity):\n")
        for rank, (doc_id, score) in enumerate(top_docs, start=1):
            print(f"{rank}. doc_id={doc_id} | similarity={score:.3f}")


if __name__ == "__main__":
    main()