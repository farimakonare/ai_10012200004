"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004
Module: chunker.py — Two chunking strategies: fixed-size (election) and semantic (budget).
"""

import os
import json
import hashlib
from typing import List, Dict, Any


PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
CHUNK_SCHEMA_VERSION = 2


def fixed_size_chunk(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Split text into fixed-size chunks with overlap. Breaks at sentence boundaries when possible."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = max(chunk.rfind(". "), chunk.rfind(".\n"), chunk.rfind("! "))
            if last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


def semantic_chunk(text: str, min_words: int = 30, max_words: int = 150) -> List[str]:
    """Split on paragraph boundaries; merge short paragraphs, split very long ones."""
    import re
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    merged: List[str] = []
    buffer = ""

    for para in raw_paras:
        word_count = len(para.split())
        if word_count < min_words:
            buffer = (buffer + " " + para).strip() if buffer else para
        else:
            if buffer:
                merged.append(buffer)
                buffer = ""
            if word_count > max_words:
                merged.extend(fixed_size_chunk(para, chunk_size=600, overlap=60))
            else:
                merged.append(para)

    if buffer:
        merged.append(buffer)

    return [c for c in merged if len(c.split()) >= 10]


def _chunk_id(text: str) -> str:
    """Deterministic content-hash ID (also aids deduplication)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def wrap_chunks(texts: List[str], base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    seen = set()
    docs = []
    for text in texts:
        cid = _chunk_id(text)
        if cid in seen:
            continue
        seen.add(cid)
        docs.append({
            "id": cid,
            "text": text,
            "metadata": {**base_metadata, "chunk_schema_version": CHUNK_SCHEMA_VERSION},
        })
    return docs


def chunk_election_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for doc in docs:
        sub_chunks = fixed_size_chunk(doc["text"], chunk_size=400, overlap=80)
        chunks.extend(wrap_chunks(sub_chunks, doc["metadata"]))
    return chunks


def chunk_budget_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for doc in docs:
        sub_chunks = semantic_chunk(doc["text"], min_words=30, max_words=150)
        if not sub_chunks:
            sub_chunks = [doc["text"]]
        chunks.extend(wrap_chunks(sub_chunks, doc["metadata"]))
    return chunks


def save_chunks(chunks: List[Dict[str, Any]], filename: str) -> str:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return path


def load_chunks(filename: str) -> List[Dict[str, Any]]:
    path = os.path.join(PROCESSED_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_and_save_all_chunks(
    election_docs: List[Dict[str, Any]],
    budget_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    election_chunks = chunk_election_documents(election_docs)
    budget_chunks = chunk_budget_documents(budget_docs)
    all_chunks = election_chunks + budget_chunks

    save_chunks(election_chunks, "election_chunks.json")
    save_chunks(budget_chunks, "budget_chunks.json")
    save_chunks(all_chunks, "all_chunks.json")

    print(f"Saved {len(election_chunks)} election chunks, "
          f"{len(budget_chunks)} budget chunks, "
          f"{len(all_chunks)} total")
    return all_chunks
