# System Architecture
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004

---

## Overview

The system runs in two phases: an offline indexing phase that runs once on startup, and an online query phase that runs on every user question. Neither phase uses an end-to-end RAG framework.

---

## Data flow

```
╔══════════════════════════════════════════════════════════════════════╗
║                     OFFLINE INDEXING PHASE                          ║
║                                                                      ║
║  Ghana Election CSV ──► csv_loader.py ──► Row -> NL sentence         ║
║                                  │                                   ║
║  2025 Budget PDF ────► pdf_loader.py ──► Page text extraction        ║
║                                  │                                   ║
║                             chunker.py                               ║
║                         ┌─────────┴──────────┐                      ║
║                Fixed-size+overlap       Semantic/paragraph           ║
║                (election rows)          (budget paragraphs)          ║
║                         └─────────┬──────────┘                      ║
║                             all_chunks.json                          ║
║                                  │                                   ║
║                           embedder.py                                ║
║                    sentence-transformers (all-MiniLM-L6-v2)          ║
║                    384-dim L2-normalized vectors                     ║
║                                  │                                   ║
║                          vector_store.py                             ║
║                         FAISS IndexFlatIP                            ║
║                         faiss.index (persisted to disk)             ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                     ONLINE QUERY PHASE                              ║
║                                                                      ║
║  User (Streamlit UI)                                                 ║
║       │ question                                                     ║
║       ▼                                                              ║
║  pipeline.py  <────────────────────── memory.py                     ║
║       │                           (last 5 Q&A turns)                ║
║       ▼                                                              ║
║  embedder.py  -> encode_query() -> 384-dim L2-normalized vector      ║
║       │                                                              ║
║       ▼                                                              ║
║  retriever.py  (HybridRetriever)                                     ║
║  ┌────────────────────────────────────────────────────┐             ║
║  │  BM25 (rank-bm25)           FAISS search           │             ║
║  │  keyword matching           cosine similarity      │             ║
║  │  weight: 0.4                weight: 0.6            │             ║
║  │                                                    │             ║
║  │  + query expansion (for vague geographic queries)  │             ║
║  └──────────────────┬─────────────────────────────────┘             ║
║                     │ top-k chunks + combined scores                 ║
║                     ▼                                                ║
║  Confidence threshold (>= 0.25)                                      ║
║  │ FAIL: return "no information" message, skip LLM call             ║
║  │ PASS: continue                                                    ║
║                     │                                                ║
║                     ▼                                                ║
║  prompt_builder.py                                                   ║
║  ┌──────────────────────────────────────────┐                        ║
║  │  Context window management               │                        ║
║  │  (truncate to <= 2000 tokens)            │                        ║
║  │  Template 3: memory + context + query    │                        ║
║  └──────────────────────────────────────────┘                        ║
║                     │ final prompt string                            ║
║                     ▼                                                ║
║  llm_client.py  -> Groq API -> llama-3.3-70b-versatile              ║
║                     │ response text                                  ║
║                     ▼                                                ║
║  logger.py -> PipelineLog (all 6 stages captured)                    ║
║                     │                                                ║
║                     ▼                                                ║
║  Streamlit UI <- PipelineResult                                      ║
║  ├── Chat response                                                   ║
║  ├── Retrieved chunks + scores                                       ║
║  ├── Final prompt (expandable)                                       ║
║  └── Stage-by-stage logs                                             ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Components

### Data ingestion

| File | What it does |
|---|---|
| `src/ingestion/csv_loader.py` | Reads the election CSV, normalizes column names, converts each row into a natural-language sentence |
| `src/ingestion/pdf_loader.py` | Extracts text page by page with pdfminer.six, strips repeated headers and footers, normalizes unicode |
| `src/ingestion/chunker.py` | Applies two chunking strategies, deduplicates by content hash, saves chunks to JSON |

### Embedding and storage

| File | What it does |
|---|---|
| `src/retrieval/embedder.py` | Singleton sentence-transformers model, returns L2-normalized 384-dim float32 vectors |
| `src/retrieval/vector_store.py` | FAISS IndexFlatIP; saves to disk after first build, loads on subsequent runs |

### Retrieval

| File | What it does |
|---|---|
| `src/retrieval/retriever.py` | BM25 (0.4) + vector (0.6) combined search; query expansion for vague queries; confidence threshold check |

### Generation

| File | What it does |
|---|---|
| `src/generation/prompt_builder.py` | Three prompt templates; context window truncation; context formatting with source tags |
| `src/generation/llm_client.py` | Groq API wrapper with exponential-backoff retry on rate limit errors |

### Orchestration

| File | What it does |
|---|---|
| `src/pipeline.py` | Runs all stages in sequence; returns PipelineResult with response, chunks, prompt, and log |
| `src/logger.py` | Captures per-stage duration and data; serialized to dict for display in UI |
| `src/memory.py` | Stores last 5 Q&A turns; formats them as a conversation history block for the prompt |

---

## Design justifications

### Why hybrid search instead of pure vector?

The election dataset has structured keywords — party abbreviations (NPP, NDC), specific constituency names, year values — where exact term matching matters. A vector-only search may rank a semantically similar but factually different chunk higher than the exact match. BM25 closes that gap. The budget PDF is the opposite: policies are described in full sentences, so semantic search is more useful there. Running both and combining the scores covers both cases.

### Why all-MiniLM-L6-v2?

It runs on CPU with no API cost, produces 384-dimensional vectors which are small enough for fast FAISS searches, and performs well on sentence similarity tasks. For a ~1,500 chunk corpus this is more than sufficient.

### Why FAISS IndexFlatIP instead of IVF or HNSW?

The corpus is small enough that exact search is fast. IndexFlatIP is deterministic and needs no training step. IVF or HNSW approximation would only matter at tens of thousands of vectors.

### Why Groq with Llama 3.3 70B?

It is free, it is fast, and the 128k context window means long prompts with multiple retrieved chunks are not a problem. The 70B parameter size gives better factual accuracy than smaller models on domain-specific questions.

### Why paragraph-aware chunking for the PDF?

The budget document is organized around distinct policy points, one or two per paragraph. Splitting at fixed character boundaries cuts across fiscal tables and policy sentences. Keeping paragraph boundaries intact means each chunk is about one thing, which makes it easier to retrieve for specific questions.

---

## Innovation: conversation memory

`ConversationMemory` keeps the last 5 Q&A pairs and injects them into the prompt as a conversation history block before the retrieved context. This lets the model resolve follow-up questions like "What about 2016?" after answering a 2020 election question.

The key design choice is where memory goes. It is added to the prompt, not prepended to the query before retrieval. Prepending to the query would bloat the query embedding with prior conversation text and dilute the retrieval signal. Keeping retrieval focused on the current question while giving the LLM the full conversation history in the prompt is a cleaner separation.
