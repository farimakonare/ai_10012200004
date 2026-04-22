"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004
Module: embedder.py — Singleton SentenceTransformer wrapper (all-MiniLM-L6-v2).
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union


_MODEL_NAME = "all-MiniLM-L6-v2"
_instance: Union["Embedder", None] = None


class Embedder:
    """Wraps SentenceTransformer; returns L2-normalized vectors for FAISS cosine search."""

    def __init__(self, model_name: str = _MODEL_NAME):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Returns shape (1, dim) for FAISS compatibility."""
        return self.encode([query])


def get_embedder() -> Embedder:
    """Return the singleton Embedder (loads model on first call)."""
    global _instance
    if _instance is None:
        _instance = Embedder()
    return _instance
