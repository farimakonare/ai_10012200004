"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004
Module: pipeline.py — Full RAG pipeline orchestration.
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .retrieval.embedder import get_embedder
from .retrieval.vector_store import VectorStore
from .retrieval.retriever import HybridRetriever
from .generation.prompt_builder import build_prompt, manage_context_window
from .generation.llm_client import generate
from .memory import ConversationMemory
from .logger import PipelineLogger, PipelineLog
from .ingestion.chunker import load_chunks, build_and_save_all_chunks, CHUNK_SCHEMA_VERSION
from .ingestion.csv_loader import load_election_documents
from .ingestion.pdf_loader import load_budget_documents


@dataclass
class PipelineResult:
    query: str
    response: str
    retrieved_chunks: List[Dict[str, Any]]
    context_used: List[Dict[str, Any]]
    prompt: str
    pipeline_log: PipelineLog
    is_relevant: bool


class RAGPipeline:
    """Full RAG pipeline. Initialize once; call .query() per user question."""

    def __init__(self, source_filter: Optional[str] = None):
        # source_filter: "election", "budget", or None (both)
        self.source_filter = source_filter
        self._embedder = None
        self._store = None
        self._retriever = None
        self._chunks: List[Dict[str, Any]] = []

    def _is_outdated_chunk_schema(self, chunks: List[Dict[str, Any]]) -> bool:
        if not chunks:
            return True
        versions = [c.get("metadata", {}).get("chunk_schema_version", 0) for c in chunks[:20]]
        return any(v < CHUNK_SCHEMA_VERSION for v in versions)

    def initialize(self, force_rebuild: bool = False) -> None:
        self._embedder = get_embedder()
        self._store = VectorStore(dim=self._embedder.dim)

        loaded_cached_index = False
        if not force_rebuild and self._store.load():
            self._chunks = self._store.chunks
            if self._is_outdated_chunk_schema(self._chunks):
                print(
                    f"Detected outdated chunk schema. Rebuilding to v{CHUNK_SCHEMA_VERSION}..."
                )
            else:
                loaded_cached_index = True

        if not loaded_cached_index:
            import os
            chunks_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "processed", "all_chunks.json"
            )
            if os.path.exists(chunks_path):
                self._chunks = load_chunks("all_chunks.json")
                if self._is_outdated_chunk_schema(self._chunks):
                    election_docs = load_election_documents()
                    budget_docs = load_budget_documents()
                    self._chunks = build_and_save_all_chunks(election_docs, budget_docs)
            else:
                print("Building chunks from raw data...")
                election_docs = load_election_documents()
                budget_docs = load_budget_documents()
                self._chunks = build_and_save_all_chunks(election_docs, budget_docs)

            if self.source_filter:
                self._chunks = [
                    c for c in self._chunks
                    if c["metadata"].get("source") == self.source_filter
                ]

            self._store.build(self._chunks, self._embedder)
            self._store.save()

        self._retriever = HybridRetriever(self._store, self._chunks)
        print(f"Pipeline ready: {self._store.size} vectors indexed.")

    def query(
        self,
        user_query: str,
        memory: Optional[ConversationMemory] = None,
        k: int = 5,
        template_id: int = 3,
    ) -> PipelineResult:
        logger = PipelineLogger(user_query)

        logger.begin_stage()
        logger.end_stage("query_received", {"query": user_query, "k": k, "template": template_id})

        logger.begin_stage()
        expanded_query = self._retriever.expand_query(user_query, self.source_filter)
        query_vec = self._embedder.encode_query(expanded_query)
        logger.end_stage("query_embedded", {
            "model": self._embedder.model_name,
            "vector_dim": int(query_vec.shape[1]),
            "embedded_query": expanded_query,
        })

        logger.begin_stage()
        results = self._retriever.retrieve(expanded_query, query_vec, k=k)
        is_relevant = self._retriever.is_relevant(results)
        logger.end_stage("retrieval", {
            "expanded_query": expanded_query,
            "num_results": len(results),
            "is_relevant": is_relevant,
            "top_scores": [
                {
                    "text_preview": r["chunk"]["text"][:80],
                    "combined": r["combined_score"],
                    "vector": r["vector_score"],
                    "bm25": r["bm25_score"],
                }
                for r in results
            ],
        })

        logger.begin_stage()
        context_used = manage_context_window(results)
        logger.end_stage("context_window", {
            "chunks_before": len(results),
            "chunks_after": len(context_used),
            "total_chars": sum(len(r["chunk"]["text"]) for r in context_used),
        })

        logger.begin_stage()
        memory_text = memory.format_for_prompt() if memory else ""

        if not is_relevant:
            response = "I don't have enough relevant information in the knowledge base to answer this question."
            prompt = "(No prompt sent — confidence below threshold)"
            logger.end_stage("prompt_skipped", {"reason": "confidence_below_threshold"})
            log = logger.done()
            return PipelineResult(
                query=user_query,
                response=response,
                retrieved_chunks=results,
                context_used=context_used,
                prompt=prompt,
                pipeline_log=log,
                is_relevant=False,
            )

        prompt = build_prompt(user_query, context_used, memory_text, template_id=template_id)
        logger.end_stage("prompt_built", {
            "template_id": template_id,
            "prompt_length_chars": len(prompt),
            "memory_turns": len(memory) if memory else 0,
            "prompt_preview": prompt[:300],
        })

        logger.begin_stage()
        response = generate(prompt)
        logger.end_stage("llm_generation", {
            "model": "llama-3.3-70b-versatile",
            "response_length_chars": len(response),
            "response_preview": response[:200],
        })

        log = logger.done()

        if memory is not None:
            memory.add_turn(user_query, response)

        return PipelineResult(
            query=user_query,
            response=response,
            retrieved_chunks=results,
            context_used=context_used,
            prompt=prompt,
            pipeline_log=log,
            is_relevant=True,
        )
