"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004
Module: pdf_loader.py — Extract and clean text from the 2025 Ghana Budget PDF.
"""

import os
import re
import unicodedata
from typing import List, Dict, Any

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


RAW_PDF = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "2025-Budget-Statement.pdf"
)

# Repeated headers, footers, and page numbers to strip
_NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"Ministry of Finance", re.IGNORECASE),
    re.compile(r"Budget Statement and Economic Policy", re.IGNORECASE),
    re.compile(r"Republic of Ghana", re.IGNORECASE),
    re.compile(r"^\s*page\s+\d+", re.IGNORECASE),
    re.compile(r"www\.mofep\.gov\.gh", re.IGNORECASE),
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\x0c", "\n")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s{3,}", "  ", text)
    return text.strip()


def _is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(p.search(stripped) for p in _NOISE_PATTERNS)


def extract_page_texts(path: str = RAW_PDF) -> List[Dict[str, Any]]:
    pages = []
    for page_num, page_layout in enumerate(extract_pages(path), start=1):
        lines = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for line in element.get_text().splitlines():
                    if not _is_noise(line):
                        lines.append(line)

        text = _normalize("\n".join(lines))
        if len(text) < 50:  # skip image/scanned pages
            continue
        pages.append({"page": page_num, "text": text})
    return pages


def to_paragraphs(page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    paragraphs = []
    for para in re.split(r"\n{2,}", page_data["text"]):
        cleaned = para.strip().replace("\n", " ")
        if len(cleaned) < 30:
            continue
        paragraphs.append({
            "text": cleaned,
            "metadata": {"source": "budget", "page": page_data["page"]},
        })
    return paragraphs


def load_budget_documents(path: str = RAW_PDF) -> List[Dict[str, Any]]:
    pages = extract_page_texts(path)
    docs = []
    for page in pages:
        docs.extend(to_paragraphs(page))
    return docs


if __name__ == "__main__":
    docs = load_budget_documents()
    print(f"Loaded {len(docs)} budget paragraphs")
    if docs:
        print("Sample:", docs[0]["text"][:200])
