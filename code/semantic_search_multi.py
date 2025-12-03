# code/semantic_search_multi.py
import json
from pathlib import Path
from collections import defaultdict

import numpy as np          # pip install numpy
from sentence_transformers import SentenceTransformer

EMBED_FILE = Path(__file__).resolve().parent.parent / "chunks_with_embeddings_all_local.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_CHUNKS = 5  # top chunks to show
TOP_K_DOCS = 5    # top docs to recommend


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


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loading embeddings from {EMBED_FILE}")
    records, matrix = load_index(EMBED_FILE)
    print(f"Loaded {len(records)} chunks")

    while True:
        query = input("\nEnter your question (or press Enter to quit): ").strip()
        if not query:
            print("Goodbye!")
            break

        # Encode query
        query_emb = model.encode([query])[0]
        sims = cosine_sim(query_emb, matrix)

        # ----- Top chunks -----
        top_k_chunks = min(TOP_K_CHUNKS, len(records))
        top_idx = np.argsort(-sims)[:top_k_chunks]

        print(f"\nTop {top_k_chunks} chunks:\n")
        for rank, idx in enumerate(top_idx, start=1):
            rec = records[idx]
            score = sims[idx]
            doc_id = rec["doc_id"]
            page = rec["page"]
            chunk_id = rec["chunk_id"]
            text = rec["text"]
            snippet = text[:400].replace("\n", " ") + ("..." if len(text) > 400 else "")

            print(f"{rank}. doc_id={doc_id} | chunk_id={chunk_id} | page={page} | score={score:.3f}")
            print(f"   {snippet}\n")

        # ----- Top documents (recommendations) -----
        doc_scores = defaultdict(list)
        for idx, rec in enumerate(records):
            doc_scores[rec["doc_id"]].append(sims[idx])

        # Aggregate: max similarity per doc
        doc_score_agg = {doc_id: max(scores) for doc_id, scores in doc_scores.items()}
        top_docs = sorted(doc_score_agg.items(), key=lambda x: x[1], reverse=True)[:TOP_K_DOCS]

        print(f"Top {len(top_docs)} documents for this query:\n")
        for rank, (doc_id, score) in enumerate(top_docs, start=1):
            print(f"{rank}. doc_id={doc_id} | score={score:.3f}")


if __name__ == "__main__":
    main()