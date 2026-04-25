"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004
Module: retriever.py — Hybrid BM25 + FAISS retriever.
combined_score = 0.4 × bm25_normalized + 0.6 × vector_score
"""

import re
import numpy as np
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi


VECTOR_WEIGHT = 0.6
BM25_WEIGHT   = 0.4
CONFIDENCE_THRESHOLD = 0.25  # scores below this trigger the no-information fallback


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


class HybridRetriever:
    """BM25 keyword search + FAISS vector search, combined by weighted score."""

    def __init__(self, vector_store, chunks: List[Dict[str, Any]]):
        self.store = vector_store
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

    def _normalize_bm25(self, scores: np.ndarray) -> np.ndarray:
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-8:
            return np.zeros_like(scores)
        return (scores - mn) / (mx - mn)

    def retrieve(
        self,
        query: str,
        query_vec: np.ndarray,
        k: int = 5,
        candidate_k: int = 20,
    ) -> List[Dict[str, Any]]:
        faiss_results = self.store.search(query_vec, k=candidate_k)
        if not faiss_results:
            return []

        faiss_score_map: Dict[str, float] = {c["id"]: s for c, s in faiss_results}

        query_tokens = _tokenize(query)
        all_bm25 = np.array(self.bm25.get_scores(query_tokens), dtype=np.float32)

        candidate_chunks = [c for c, _ in faiss_results]
        candidate_indices = [
            next(i for i, ch in enumerate(self.chunks) if ch["id"] == c["id"])
            for c in candidate_chunks
        ]
        normalized_bm25 = self._normalize_bm25(all_bm25[candidate_indices])

        results = []
        for i, chunk in enumerate(candidate_chunks):
            v_score = faiss_score_map[chunk["id"]]
            b_score = float(normalized_bm25[i])
            combined = VECTOR_WEIGHT * v_score + BM25_WEIGHT * b_score
            results.append({
                "chunk": chunk,
                "vector_score": round(v_score, 4),
                "bm25_score":   round(b_score, 4),
                "combined_score": round(combined, 4),
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:k]

    def is_relevant(self, results: List[Dict[str, Any]]) -> bool:
        if not results:
            return False
        return results[0]["combined_score"] >= CONFIDENCE_THRESHOLD

    def expand_query(self, query: str, source_filter: str = None) -> str:
        """
        Expand vague geographic queries with domain context.
        Specific queries (containing year, candidate, budget terms) are left unchanged
        to avoid diluting the embedding.
        """
        north_pattern    = re.compile(r"\b(north|northern|upper)\b", re.IGNORECASE)
        winner_pattern   = re.compile(r"\b(who won|winner|won the .* election|election winner)\b", re.IGNORECASE)
        budget_keywords  = re.compile(r"\b(budget|fiscal|revenue|expenditure|policy|mofep)\b", re.I)
        election_keywords = re.compile(r"\b(election|vote|votes|party|candidate|won|win|npp|ndc)\b", re.I)

        expansions = []

        if north_pattern.search(query):
            if source_filter == "election" or (source_filter is None and not budget_keywords.search(query)):
                expansions.append(
                    "Ghana election northern region north east region savannah region upper east region upper west region"
                )

        if winner_pattern.search(query):
            expansions.append("Ghana presidential election national summary winner total votes")

        if source_filter == "budget" and not election_keywords.search(query):
            expansions.append("2025 Ghana budget")

        if source_filter == "election" and not budget_keywords.search(query):
            expansions.append("Ghana election")

        expanded = " ".join(expansions) + " " + query if expansions else query
        return expanded.strip()
