import json
import re
from pathlib import Path

from PyPDF2 import PdfReader  # pip install PyPDF2

from config_docs import DOCS, BASE_DIR


def clean_markdown(md: str) -> str:
    """
    Clean the raw markdown from the OCR JSON.
    - Remove <a id='...'></a> anchors
    - Remove <:: ... ::> metadata blocks
    - Remove HTML comments except PAGE BREAK markers
    """
    if not isinstance(md, str):
        return ""

    md = re.sub(r"<a id='[^']*'></a>", "", md)
    md = re.sub(r"<::.*?::>", "", md, flags=re.DOTALL)
    md = re.sub(r"<!--(?! PAGE BREAK).*?-->", "", md, flags=re.DOTALL)

    return md


def clean_page_text(page: str) -> str:
    """
    Clean a single page's text:
    - Remove page labels like 'Page 3' or 'Page i'
    - Remove footnote numbers and superscript markers
    - Remove extra line breaks and collapse spaces
    """
    text = page.strip()

    text = re.sub(r"\bPage\s+[0-9ivxlcdmIVXLCDM]+\b", "", text)
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\^\d+", "", text)
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def process_from_ocr_json(doc_id: str, ocr_json_path: Path, output_dir: Path):
    """
    For docs that already went through OCR and are stored as JSON with 'markdown'
    and <!-- PAGE BREAK --> markers.
    """
    print(f"[OCR JSON] Processing {doc_id} from {ocr_json_path}")

    with ocr_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "markdown" not in data:
        raise ValueError(f"{ocr_json_path} does not contain a 'markdown' field.")

    full_md = data["markdown"]
    cleaned_md = clean_markdown(full_md)
    pages = cleaned_md.split("<!-- PAGE BREAK -->")

    cleaned_pages = []
    for i, page in enumerate(pages):
        page_text = clean_page_text(page)
        if not page_text:
            continue
        cleaned_pages.append(f"--- Page {i} ---\n{page_text}")

    final_text = "\n\n".join(cleaned_pages)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{doc_id}_clean_final.txt"
    out_path.write_text(final_text, encoding="utf-8")

    print(f"  → wrote {out_path}")


def process_from_pdf(doc_id: str, pdf_path: Path, output_dir: Path):
    """
    For readable PDFs (non-OCR JSON):
    - Extract text page by page using PyPDF2
    - Clean each page
    - Write {doc_id}_clean_final.txt with --- Page i --- markers

    NOTE: If the PDF is actually scanned images (no real text),
    this will produce empty output. Those need OCR instead.
    """
    print(f"[PDF] Processing {doc_id} from {pdf_path}")

    reader = PdfReader(str(pdf_path))
    cleaned_pages = []

    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        page_text = clean_page_text(raw_text)
        if not page_text:
            # Empty or image-only page
            continue
        cleaned_pages.append(f"--- Page {i} ---\n{page_text}")

    final_text = "\n\n".join(cleaned_pages)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{doc_id}_clean_final.txt"
    out_path.write_text(final_text, encoding="utf-8")

    print(f"  → wrote {out_path}")


def main():
    output_dir = BASE_DIR / "clean_text"

    for doc in DOCS:
        doc_id = doc["doc_id"]
        doc_type = doc.get("type", "pdf")

        if doc_type == "ocr_json":
            ocr_json_path = Path(doc["ocr_json"])
            process_from_ocr_json(doc_id, ocr_json_path, output_dir)
        elif doc_type == "pdf":
            pdf_path = Path(doc["pdf_path"])
            process_from_pdf(doc_id, pdf_path, output_dir)
        else:
            print(f"⚠️ Unknown type '{doc_type}' for {doc_id}, skipping.")


if __name__ == "__main__":
    main()