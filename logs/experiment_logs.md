# Experiment logs
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004

---

## Log 1 — Chunking strategy comparison

**Date:** 2026-04-25  
**Notebook:** `experiments/01_chunking_analysis.ipynb`

### Raw notebook outputs
- Election docs: **623**
- Budget docs (paragraphs): **251**
- Election chunks: **623**
- Election chunk length: mean **137** chars, min **113**, max **177**
- Budget chunks: **949**
- Budget chunk length: mean **444** chars, min **51**, max **1010**
- Saved: **623 election + 949 budget = 1572 total chunks**

### Sample chunks seen
- Election sample:  
  `In the 2020 Ghana presidential election, Nana Akufo Addo of the NPP party received 145584 votes (55.04%) in Ahafo Region (formerly part of Brong Ahafo Region).`
- Budget sample:  
  `THEME: Resetting The Economy For The Ghana We Want`

### Observation
- Election chunks remain short, consistent, and row-fact oriented.
- Budget chunks remain paragraph-like with wider size variance.
- The updated election pipeline now includes normalized region wording and expanded election records in the processed set.

---

## Log 2 — Retrieval failure case and fix

**Date:** 2026-04-25  
**Notebook:** `experiments/02_retrieval_failures.ipynb`  
**Query:** `"What happened in the north?"`

### Before fix (no query expansion)
1. `combined=0.497`, source=election  
   `In the 2020 Ghana presidential election, Nana Konadu Agyeman Rawlings of the NDP party ...`
2. `combined=0.496`, source=election  
   `In the 2020 Ghana presidential election, Nana Konadu Agyeman Rawlings of the NDP party ...`
3. `combined=0.450`, source=budget  
   `... North-East, Oti, Savannah, and Western North Regions ...`

### After fix (with query expansion)
Expanded query used:
`Ghana election northern region north east region savannah region upper east region upper west region Ghana election What happened in the north?`

1. `combined=0.733`, source=election  
   `In the 2020 Ghana presidential election, Akua Donkor of the GFP party ... North East Region ...`
2. `combined=0.723`, source=election  
   `In the 2020 Ghana presidential election, John Dramani Mahama ... North East Region ...`
3. `combined=0.722`, source=election  
   `In the 2016 Ghana presidential election, John Dramani Mahama ... North East Region ...`

### OOD check
Query: `"What is the capital of France?"`
- `is_relevant`: **True** (threshold 0.25)
- Top scores: **0.525**, **0.503**, **0.454**
- Retrieved context remained budget/economic chunks, not France knowledge.

### Interpretation
- Query expansion clearly improved north-focused retrieval relevance.
- OOD gating is still weak at retrieval score level (false positive relevance persists).

---

## Log 3 — Prompt template comparison

**Date:** 2026-04-25  
**Notebook:** `experiments/03_prompt_comparison.ipynb`  
**Query:** `"Who won the 2020 presidential election in Ghana?"`

### Template outputs
- **Template 1 (Baseline):**  
  `The provided context does not contain information about the overall winner ... only vote counts in certain regions.`

- **Template 2 (Hallucination-Controlled):**  
  `I don't have enough information in the knowledge base to answer this question.`

- **Template 3 (With Memory):**  
  `I cannot determine the answer ... context only mentions vote counts for Christian Kwabena Andrews (GUM) ... missing major candidates and national outcome.`

### Context window output
- Chunks before: **5**
- Chunks after: **5**
- Total context chars: **642**

### Interpretation
- In this notebook run, all templates were conservative and refused a winner claim.
- Template 2 remained the strictest hallucination-control behavior.
- Template 3 gave the most explanatory refusal.

---

## Log 4 — Adversarial tests (RAG vs pure LLM)

**Date:** 2026-04-25  
**Notebook:** `experiments/04_adversarial_tests.ipynb`

### Query 1: `"Who won?"` (ambiguous)

**RAG response summary**
- Returned year-wise winners from retrieved national summaries.
- Included an inconsistency statement for 2008 ("context says Nana Addo as winner, but actual winner J. A. Mills"), which is outside strict context-only behavior.
- `is_relevant`: **True**
- Top chunk: 2016 national summary (`Nana Akufo Addo (NPP) ...`)

**Pure LLM response summary**
- Asked for clarification and did not pick a specific event or winner.

**Interpretation**
- RAG was more specific but showed leakage risk (external correction language).
- Pure LLM stayed cautious but less useful.

---

### Query 2: `"What did the president say about free healthcare in the 2025 budget?"`

**RAG response summary**
- Grounded answer citing:
  - `Free Primary Healthcare – We are delivering!`
  - `MahamaCares ... non-communicable diseases`
- `is_relevant`: **True**
- Top chunk score: **0.6567**

**Pure LLM response summary**
- Generic refusal based on knowledge cutoff; no grounded citation.

**Interpretation**
- For grounded in-domain budget questions, RAG outperformed pure LLM usefulness.

---

## Log 5 — Multi-turn memory behavior (UI run)

**Date:** 2026-04-25  
**Method:** Streamlit UI conversation run

Conversation sequence included:
1. `Who won the 2020 presidential election in Ghana?` -> grounded national winner answer.
2. `Which party dominated the Ashanti Region?` -> NPP answer from retrieved regional chunks (with caveat).
3. `What is the total expenditure in the 2025 budget?` -> GH¢268.8 billion.
4. `How did NDC perform in Northern Ghana in 2016?` -> regional percentage summary with caveat.
5. `Who won?` -> context-influenced multi-year answer.
6. `What is the capital of France?` -> refusal based on irrelevant retrieved context.
7. `What did the president say about free healthcare ... ?` -> grounded budget quote answer.

### Interpretation
- Memory is active and influences follow-up behavior.
- Helps continuity, but can bias underspecified prompts like `"Who won?"`.

---

## Updated summary (current evidence)

| Experiment | Key finding | Current status |
|---|---|---|
| Chunking (Part A) | Updated pipeline produced 1572 chunks (623 election, 949 budget) with expected distribution | Stable |
| Retrieval failure + fix (Part B) | Query expansion improved north-query relevance significantly | Improved, but not perfect |
| OOD retrieval (Part B) | `capital of France` still passes score threshold (`is_relevant=True`) | Known limitation |
| Prompt comparison (Part C) | All templates still conservative on winner query in notebook run | Hallucination control strong, directness variable |
| Adversarial comparison (Part E) | RAG gives grounded value on budget query; ambiguous query still risky | Partially improved |
| Innovation / memory (Part G) | Multi-turn continuity works, but can bias ambiguous follow-ups | Working with tradeoff |
