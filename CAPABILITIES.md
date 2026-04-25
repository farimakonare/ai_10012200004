# Ghana Knowledge Assistant — Capabilities Overview

**Student:** Farima Konaré | **Index:** 10012200004 | **Course:** CS4241, Academic City University

---

## What This Project Does

The Ghana Knowledge Assistant is a conversational AI system that answers questions about two specific topics:

1. **Ghana Presidential Election Results** — candidate names, party vote counts, percentages, and regional breakdowns across multiple election years
2. **Ghana 2025 National Budget** — revenue figures, expenditure targets, sector allocations, fiscal policy decisions, and economic projections

When you ask a question, the system first finds the most relevant passages from these datasets, then uses a large language model to write a grounded answer based only on what was retrieved. It does not rely on the model's general training data for specific facts.

---

## The Problem It Solves

Large language models hallucinate — they confidently state facts that are wrong. This is especially dangerous for specific numerical data like election results and budget figures, where the model may have outdated or simply invented statistics.

This system addresses that by always grounding the answer in retrieved source passages. The model only states what the documents actually say. If the answer is not in the dataset, the system says so explicitly rather than guessing.

---

## Full Feature List

### Hybrid Search Retrieval
Every question triggers two independent searches that run in parallel:
- **Keyword search (BM25)** — finds passages containing exact words from the question. This is strong for proper nouns like party names (NPP, NDC), constituency names, and specific numbers.
- **Vector search (FAISS)** — finds passages that are semantically similar even if they use different words. This is strong for paraphrased or conceptual questions.

The final score is a weighted combination: 40% BM25 + 60% vector. Chunks are ranked by this combined score and the top results are sent to the model.

### Confidence Thresholding
If the highest-scoring retrieved chunk scores below 0.25 out of 1.0, the system determines the question cannot be reliably answered from the knowledge base. It returns a clear "I don't have enough information" message instead of calling the language model at all. This reduces hallucination risk, but score thresholding alone does not fully solve out-of-domain detection.

### Conversation Memory
The assistant remembers the last five exchanges in a conversation. This lets you ask follow-up questions naturally:

> "Who won the 2020 election?"
> "What about 2016?" ← the assistant knows you mean the election

Memory is injected as conversation history in the prompt and is cleared when you click "Clear conversation" in the sidebar.

### Query Expansion
For vague geographic questions like "What happened in the north?", the system automatically prepends domain context ("Ghana election northern region north east region savannah region upper east region upper west region") before embedding. For winner-style queries, it also adds national-summary intent terms to improve retrieval of election winner records. Specific questions (those already containing clear year/party/candidate signals) are expanded minimally to avoid diluting retrieval.

### Context Window Management
Retrieved passages are ranked by score and included in the prompt until the total character count approaches the token budget (~8,000 characters, approximately 2,000 tokens). Lower-scoring chunks are dropped first. This ensures the prompt always fits within the model's context window without truncating mid-sentence.

### Three Prompt Templates
You can switch between three prompt strategies in Advanced Settings:

| Template | Behaviour |
|---|---|
| Baseline | Pass context and question, ask for an answer |
| Hallucination-controlled | Explicit instruction not to guess; fall back message if insufficient context |
| With memory (default) | Includes the last five conversation turns; same hallucination guardrails |

### Source Filtering
Use the segmented control in the sidebar to restrict retrieval to one dataset:
- **Both** — searches election and budget chunks together
- **Elections** — only Ghana election result data
- **Budget** — only 2025 national budget data

Filtering is useful when a question could be ambiguous across both datasets.

### Transparent Pipeline Logs
Below every response, you can expand three panels:
- **Retrieved chunks** — the exact passages used, with source tag (ELECTION/BUDGET), combined score, vector score, and BM25 score displayed as color-coded bars
- **Final prompt** — the exact text sent to the language model, so you can verify what the model saw
- **Pipeline log** — per-stage timing for query embedding, retrieval, context management, prompt construction, and generation

### Structured Data Ingestion
- **Election CSV** — each row is converted into a natural-language sentence: *"In the 2020 Ghana presidential election, Nana Akufo-Addo of the NPP party received 145,584 votes (55.04%) in Brong Ahafo Region."* This format embeds significantly better than raw key-value pairs.
- **Election national summaries** — one synthetic summary record is added per election year (winner, runner-up, and total votes) so winner-style questions can retrieve direct national evidence.
- **Budget PDF** — text is extracted page by page using pdfminer.six. Headers, footers, page numbers, and repeated boilerplate lines are stripped. The remaining text is split on paragraph boundaries, preserving the document's logical structure.

### Two Chunking Strategies
- **Fixed-size with overlap** (400 characters, 80-character overlap) — used for election data. Overlap prevents key sentences from being cut at a chunk boundary.
- **Semantic / paragraph-aware** — used for the budget PDF. Paragraphs shorter than 30 words are merged; paragraphs longer than 150 words are split. This preserves the budget document's natural logical units.

### Persistent FAISS Index
On first startup the system builds a FAISS vector index over about 1,572 chunks and saves it to disk. On every subsequent startup it loads from disk in under two seconds. Rebuilding only happens if you explicitly clear the processed data (or when the chunk schema version changes).

---

## Sample Questions the System Handles Well

**Election questions**
- Who won the 2020 presidential election in Ghana?
- How many votes did John Mahama receive in the Volta Region in 2016?
- Which party dominated the Ashanti Region across elections?
- Compare NDC and NPP performance in Northern Ghana

**Budget questions**
- What is the 2025 total government expenditure?
- What is the fiscal deficit target for 2025?
- How much is allocated to the education sector?
- What are the main revenue sources in the 2025 budget?

**Multi-turn (memory)**
- "Who won 2020?" → "What about 2016?" → "And the Ashanti region specifically?"

---

## What the System Cannot Do

- Answer questions outside the two datasets (no general Ghana history, no current news)
- Provide data from elections before the dataset's coverage range
- Perform arithmetic (e.g., "total votes across all regions") — it retrieves text, not a database query engine
- Give real-time or post-2025 information

For out-of-domain questions, the system relies on a combination of retrieval scoring and prompt guardrails to avoid unsupported answers.
