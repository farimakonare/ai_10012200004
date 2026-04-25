# Ghana Knowledge Assistant
CS4241 — Introduction to Artificial Intelligence | End-of-Semester Exam
Student: Farima Konaré | Index: 10012200004
Lecturer: Godwin N. Danso | Academic City University

---

## What it does

Answers questions about Ghana's presidential election results (2000–2020) and the 2025 national budget. Built as a custom RAG pipeline — no LangChain, no LlamaIndex. Chunking, retrieval, and prompt construction are all hand-coded.

**Live demo:** [Add Streamlit Cloud URL after deployment]

---

## How it works

```
User question
    -> Query expansion (vague geographic queries get domain context added)
    -> Hybrid retrieval: BM25 keyword + FAISS vector search, weighted 40/60
    -> Confidence check: if best score < 0.25, skip the LLM entirely
    -> Context window: top chunks truncated to ≤2000 tokens
    -> Prompt built with last 5 conversation turns injected as memory
    -> Groq / Llama 3.3 70B generates a grounded answer
    -> Every stage logged and visible in the UI
```

---

## Setup

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the datasets
```bash
python scripts/download_data.py
```
Pulls the Ghana election CSV and 2025 budget PDF into `data/raw/`.

### 3. Set your Groq API key
Get a free key at console.groq.com, then create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_key_here"
```

### 4. Run
```bash
streamlit run app.py
```
First run builds the FAISS index over ~1,572 chunks — takes about a minute. Loads from disk after that.

---

## Project structure

```
├── app.py                     # Streamlit chat UI
├── requirements.txt
├── CAPABILITIES.md            # Feature overview in plain language
├── data/
│   ├── raw/                   # Downloaded datasets (gitignored)
│   └── processed/             # FAISS index + chunk cache (gitignored)
├── src/
│   ├── ingestion/             # CSV and PDF loading, chunking
│   ├── retrieval/             # Embedder, FAISS store, hybrid retriever
│   ├── generation/            # Prompt templates, Groq client
│   ├── pipeline.py            # End-to-end orchestration
│   ├── memory.py              # Conversation memory
│   └── logger.py              # Per-stage timing logs
├── experiments/               # Jupyter notebooks (Parts A, B, C, E)
├── logs/
│   └── experiment_logs.md     # Manual experiment observations
└── docs/
    └── architecture.md        # System design and justifications
```

---

## Design decisions

| Component | Choice | Reason |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 | Free, local, 384-dim, good sentence similarity |
| Vector store | FAISS IndexFlatIP | Exact cosine search, no server needed |
| LLM | Groq / Llama 3.3 70B | Free tier, fast, 128k context window |
| Retrieval | Hybrid BM25 + Vector | BM25 catches exact party and constituency names; vector handles paraphrased budget queries |
| Innovation | Conversation memory | Follow-up questions work without restating context |

---

## Running the experiment notebooks

```bash
cd experiments
jupyter notebook
```

Run in order: `01_chunking_analysis.ipynb` → `02_retrieval_failures.ipynb` → `03_prompt_comparison.ipynb` → `04_adversarial_tests.ipynb`

Record observations in `logs/experiment_logs.md` while the notebooks run.

---
