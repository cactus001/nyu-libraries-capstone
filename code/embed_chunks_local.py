# code/embed_chunks_local.py
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer  # pip install sentence-transformers torch

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_CHUNKS = BASE_DIR / "chunks_token_based_all.jsonl"
OUTPUT_EMBEDS = BASE_DIR / "chunks_with_embeddings_all_local.jsonl"

# Strong retrieval model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"#"sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 64


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    if not INPUT_CHUNKS.exists():
        raise FileNotFoundError(f"Could not find input chunks file at: {INPUT_CHUNKS}")

    with INPUT_CHUNKS.open("r", encoding="utf-8") as fin, \
         OUTPUT_EMBEDS.open("w", encoding="utf-8") as fout:

        batch_records = []
        batch_texts = []

        for line in fin:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            batch_records.append(rec)
            batch_texts.append(rec["text"])

            if len(batch_records) >= BATCH_SIZE:
                process_batch(model, batch_records, batch_texts, fout)
                batch_records = []
                batch_texts = []

        if batch_records:
            process_batch(model, batch_records, batch_texts, fout)

    print(f"Done! Wrote embeddings to {OUTPUT_EMBEDS}")


def process_batch(model, records, texts, fout):
    embeddings = model.encode(texts, show_progress_bar=False)

    for rec, emb in zip(records, embeddings):
        out_record = {
            "doc_id": rec["doc_id"],
            "chunk_id": rec["chunk_id"],
            "page": rec["page"],
            "text": rec["text"],
            "embedding": emb.tolist(),  # numpy array -> list
        }
        # Preserve document_title if it exists
        if "document_title" in rec:
            out_record["document_title"] = rec["document_title"]
        fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print("Reading chunks from:", INPUT_CHUNKS)
    print("Writing embeddings to:", OUTPUT_EMBEDS)
    main()