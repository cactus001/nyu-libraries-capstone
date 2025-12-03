from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project/
DOCS_DIR = BASE_DIR / "docs"

DOCS = [
    {
        "doc_id": "foreign_charities_1995",
        "type": "ocr_json",
        "ocr_json": DOCS_DIR / "foreign_charities_ocr.json",
    },
    {
        "doc_id": "jimaging",
        "type": "pdf",
        "pdf_path": DOCS_DIR / "jimaging.pdf",
    },
    {
        "doc_id": "managing-sharing-publishing-data",
        "type": "pdf",
        "pdf_path": DOCS_DIR / "managing-sharing-publishing-data.pdf",
    }
    # add more docs here...
]