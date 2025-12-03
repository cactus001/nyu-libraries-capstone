# code/multi_chunk_token_based.py
import re
import json
from pathlib import Path

import tiktoken  # pip install tiktoken

from config_docs import DOCS, BASE_DIR


ENCODING_NAME = "cl100k_base"
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 200

CLEAN_DIR = BASE_DIR / "clean_text"
OUTPUT_CHUNKS = BASE_DIR / "chunks_token_based_all.jsonl"


def load_pages_with_numbers(text: str):
    """
    Parse text with sections like:

    --- Page 0 ---
    ...
    --- Page 1 ---
    ...

    Returns list of dicts: [{ "page": 0, "text": "..."}, ...]
    """
    pattern = r"--- Page (\d+) ---\s*"
    parts = re.split(pattern, text)

    pages = []
    for i in range(1, len(parts), 2):
        page_num_str = parts[i]
        page_text = parts[i + 1] if i + 1 < len(parts) else ""
        page_num = int(page_num_str)
        cleaned_text = page_text.strip()
        if cleaned_text:
            pages.append({"page": page_num, "text": cleaned_text})
    return pages


def chunk_page_text(page_text: str, enc):
    """
    Token-based chunking for a single page.
    Returns list of token-sequences (list[int]).
    """
    tokens = enc.encode(page_text)
    chunks = []
    n_tokens = len(tokens)
    i = 0

    while i < n_tokens:
        chunk_token_ids = tokens[i: i + CHUNK_SIZE_TOKENS]
        if not chunk_token_ids:
            break
        chunks.append(chunk_token_ids)
        i += CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS

    return chunks


def main():
    enc = tiktoken.get_encoding(ENCODING_NAME)

    with OUTPUT_CHUNKS.open("w", encoding="utf-8") as fout:
        for doc in DOCS:
            doc_id = doc["doc_id"]
            clean_path = CLEAN_DIR / f"{doc_id}_clean_final.txt"
            print(f"Chunking {doc_id} from {clean_path}")

            if not clean_path.exists():
                print(f"  ⚠️ Clean text not found for {doc_id}, skipping.")
                continue

            full_text = clean_path.read_text(encoding="utf-8")
            pages = load_pages_with_numbers(full_text)

            global_chunk_index = 0

            for page in pages:
                page_num = page["page"]
                page_text = page["text"]

                token_chunks = chunk_page_text(page_text, enc)

                for tokens in token_chunks:
                    chunk_text = enc.decode(tokens)
                    record = {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_chunk_{global_chunk_index:04d}",
                        "page": page_num,
                        "text": chunk_text,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    global_chunk_index += 1

    print(f"Done. Wrote all chunks to {OUTPUT_CHUNKS}")


if __name__ == "__main__":
    main()