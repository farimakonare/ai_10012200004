"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004
Module: vector_store.py — FAISS IndexFlatIP vector store with save/load support.
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Dict, Any


INDEX_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
INDEX_FILE  = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.pkl")


class VectorStore:
    """
    FAISS IndexFlatIP over L2-normalized vectors.
    Inner product on unit vectors equals cosine similarity, so scores are in [-1, 1].
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.chunks: List[Dict[str, Any]] = []

    def build(self, chunks: List[Dict[str, Any]], embedder) -> None:
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        print(f"Embedding {len(texts)} chunks...")
        vecs = embedder.encode(texts, show_progress=True)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)
        print(f"FAISS index built: {self.index.ntotal} vectors")

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)
        return [
            (self.chunks[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def save(self) -> None:
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_FILE)
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"Index saved: {self.index.ntotal} vectors -> {INDEX_FILE}")

    def load(self) -> bool:
        if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
            return False
        self.index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Index loaded: {self.index.ntotal} vectors from {INDEX_FILE}")
        return True

    @property
    def size(self) -> int:
        return self.index.ntotal


def get_or_build_store(chunks: List[Dict[str, Any]] = None, embedder=None) -> "VectorStore":
    from .embedder import get_embedder
    store = VectorStore()
    if store.load():
        return store
    if chunks is None or embedder is None:
        raise ValueError("No saved index found. Provide chunks and embedder to build one.")
    emb = embedder or get_embedder()
    store.build(chunks, emb)
    store.save()
    return store
