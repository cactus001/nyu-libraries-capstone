# code/multi_chunk_token_based.py
import re
import json
from pathlib import Path
from PyPDF2 import PdfReader

import tiktoken  # pip install tiktoken

from config_docs import DOCS, BASE_DIR

ENCODING_NAME = "cl100k_base"

# More focused chunks
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 128

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


#def extract_document_title(first_page_text: str) -> str:
    """
    Extract document title from the first page text.
    Takes the first 300 characters and cleans it up to form a title.
    """
#    if not first_page_text:
#        return ""
    
    # Take first 300 characters
#   title_candidate = first_page_text[:300].strip()
    
    # Try to find a good title by looking for the first sentence or first few lines
    # Split by common separators and take the first meaningful part
#   lines = title_candidate.split('\n')
#    if lines:
        # Take first non-empty line, or first 200 chars if single line
#        first_line = lines[0].strip()
#        if first_line and len(first_line) > 10:
            # Clean up: remove extra whitespace, limit length
#            title = re.sub(r'\s+', ' ', first_line)
            # Take up to 200 characters, but try to end at a word boundary
#            if len(title) > 200:
#                title = title[:200].rsplit(' ', 1)[0]
#            return title
    
    # Fallback: just take first 200 chars and clean
#    title = re.sub(r'\s+', ' ', title_candidate)
#    if len(title) > 200:
#        title = title[:200].rsplit(' ', 1)[0]
#    return title

def extract_title_from_pdf(pdf_path: Path) -> str:
    """
    Extract title directly from the first page of the PDF.
    Special handling for patents and academic papers.
    """
    try:
        reader = PdfReader(str(pdf_path))
        if len(reader.pages) == 0:
            return ""
        
        # Get first page raw text
        first_page = reader.pages[0].extract_text() or ""
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', first_page).strip()
        
        # Special case: Patent documents
        if 'patent' in text.lower()[:500]:
            # Look for patent title after common headers
            patent_match = re.search(
                r'(?:Patent|Application|Appeal)\s+(?:No\.|Number|#)?\s*[\d\-/,]+\s+(.+?)(?:Abstract|Background|Summary|Claims|Description)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if patent_match:
                title = patent_match.group(1).strip()
                # Clean up and limit length
                title = re.sub(r'\s+', ' ', title)
                if len(title) > 150:
                    title = title[:150].rsplit(' ', 1)[0] + '...'
                return title
        
        # Special case: Academic papers (often have clear title patterns)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        # Strategy 1: Find first ALL CAPS line (common for titles)
        for i, line in enumerate(lines[:15]):
            if len(line) < 10 or len(line) > 200:
                continue
            # Check if mostly uppercase (at least 70% caps)
            alpha_chars = [c for c in line if c.isalpha()]
            if alpha_chars:
                caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
                if caps_ratio > 0.7 and not re.match(r'^(PAGE|VOLUME|CHAPTER)', line, re.I):
                    return line
        
        # Strategy 2: Take first substantial line
        for line in lines[:10]:
            if 15 <= len(line) <= 150:
                # Skip obvious headers
                if not re.match(r'^(page|volume|chapter|section|table of contents)', line, re.I):
                    return line
        
        # Fallback: first 100 chars
        return text[:100].rsplit(' ', 1)[0] if len(text) > 100 else text
        
    except Exception as e:
        print(f"  ⚠️ Error extracting title from PDF: {e}")
        return ""

def chunk_page_text(page_text: str, enc):
    """
    Token-based chunking for a single page.
    Returns list of token-id sequences.
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


#def main():
#    enc = tiktoken.get_encoding(ENCODING_NAME)

#    with OUTPUT_CHUNKS.open("w", encoding="utf-8") as fout:
#        for doc in DOCS:
#            doc_id = doc["doc_id"]
#            clean_path = CLEAN_DIR / f"{doc_id}_clean_final.txt"
#            print(f"Chunking {doc_id} from {clean_path}")

#            if not clean_path.exists():
#                print(f"  ⚠️ Clean text not found for {doc_id}, skipping.")
#                continue

#            full_text = clean_path.read_text(encoding="utf-8")
#            pages = load_pages_with_numbers(full_text)

            # Extract document title from first page (Page 0)
#            document_title = ""
#            if pages and pages[0]["page"] == 0:
#                document_title = extract_document_title(pages[0]["text"])

#            global_chunk_index = 0

#            for page in pages:
#                page_num = page["page"]
#                page_text = page["text"]

#                token_chunks = chunk_page_text(page_text, enc)

#                for tokens in token_chunks:
#                    chunk_text = enc.decode(tokens)
#                    record = {
#                        "doc_id": doc_id,
#                        "chunk_id": f"{doc_id}_chunk_{global_chunk_index:04d}",
#                        "page": page_num,
#                        "text": chunk_text,
#                        "document_title": document_title,
#                    }
#                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
#                    global_chunk_index += 1

#    print(f"Done. Wrote all chunks to {OUTPUT_CHUNKS}")

def main():
    enc = tiktoken.get_encoding(ENCODING_NAME)

    with OUTPUT_CHUNKS.open("w", encoding="utf-8") as fout:
        for doc in DOCS:
            doc_id = doc["doc_id"]
            pdf_path = doc.get("pdf_path")  # Get PDF path
            clean_path = CLEAN_DIR / f"{doc_id}_clean_final.txt"
            print(f"Chunking {doc_id} from {clean_path}")

            if not clean_path.exists():
                print(f"  ⚠️ Clean text not found for {doc_id}, skipping.")
                continue

            # Extract title from ORIGINAL PDF (not cleaned text)
            document_title = ""
            if pdf_path and Path(pdf_path).exists():
                document_title = extract_title_from_pdf(Path(pdf_path))
                print(f"  📄 Title: {document_title[:80]}...")
            
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
                        "document_title": document_title,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    global_chunk_index += 1

    print(f"Done. Wrote all chunks to {OUTPUT_CHUNKS}")

if __name__ == "__main__":
    main()