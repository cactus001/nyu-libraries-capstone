# Capstone OCR - Semantic Document Search

A semantic search system for OCR-processed documents that enables natural language queries over a large document corpus.

## Features

- **Document Processing**: Extracts and cleans text from PDFs and OCR-processed documents
- **Intelligent Chunking**: Token-based chunking with overlap to preserve context
- **Semantic Search**: Two-stage retrieval system using bi-encoder and cross-encoder models
- **Web Interface**: Modern, responsive UI for searching documents

## Setup

1. Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

2. Process documents (if not already done):
```bash
# Generate config from PDFs in docs/ folder
python3 code/generate_configs_from_docs.py

# Process documents and extract text
python3 code/multi_process_docs.py

# Chunk the text
python3 code/multi_chunk_token_based.py

# Generate embeddings
python3 code/embed_chunks_local.py
```

## Running the Web Application

Start the Flask web server:

```bash
python3 code/app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Enter your search query in natural language (e.g., "How do I configure authentication?")
2. Click "Search" or press Enter
3. View the top matching chunks and recommended documents

## Project Structure

- `code/` - Python scripts for processing and search
  - `app.py` - Flask web application
  - `multi_process_docs.py` - Document text extraction
  - `multi_chunk_token_based.py` - Text chunking
  - `embed_chunks_local.py` - Embedding generation
  - `semantic_search_multi.py` - Command-line search interface
- `docs/` - Source PDF documents
- `clean_text/` - Processed and cleaned text files
- `chunks_token_based_all.jsonl` - Chunked text data
- `chunks_with_embeddings_all_local.jsonl` - Chunks with embeddings
