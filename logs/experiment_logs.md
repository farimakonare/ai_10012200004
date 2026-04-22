# Experiment logs
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004

---

## Log 1 — Chunking strategy comparison

**Date:** 2026-04-21
**Notebook:** `experiments/01_chunking_analysis.ipynb`

### What the notebook produced
- Election CSV: 615 chunks, fixed-size strategy (400 chars, 80 overlap)
- Budget PDF: 950 chunks, paragraph-aware strategy
- Mean chunk length - election: 126 chars, budget: 443 chars
- Election chunk range: min 113, max 141
- Budget chunk range: min 51, max 1010
- Total chunks saved: 1,565

### What I observed
- Election chunk lengths are tightly clustered around ~120–135 characters with a narrow spread, showing consistent row-based chunking.
- Budget chunk lengths are much wider, with a main concentration around ~450–620 characters and a smaller short-chunk group around ~130–220, plus a rare long outlier near ~1000.
- This confirms fixed-size chunking gives stable chunk lengths for election data, while paragraph-aware chunking preserves variable policy-paragraph structure in the budget data.

### Why the strategies work
Fixed-size works for election rows because each row already has a short structured fact pattern (year, candidate, party, votes, region). Paragraph-aware works better for the budget because policy meaning is carried at paragraph level and longer chunks keep that context intact.

---

## Log 2 — Retrieval failure case

**Date:** 2026-04-21
**Notebook:** `experiments/02_retrieval_failures.ipynb`
**Query tested:** "What happened in the north?"

Before query expansion:
- Top chunk combined score: 0.555
- Top chunk source: budget
- Top chunk text (first 80 chars): `rth-East, Oti, Savannah, and Western North Regions; and  Establish one new Mi...`
- Relevant? No for intended election intent (top hit was budget text)

After query expansion ("Ghana election northern region upper east upper west Ghana election What happened in the north?"):
- Top chunk combined score: 0.761
- Top chunk source: election
- Top chunk text (first 80 chars): `In the 1992 Ghana presidential election, Others of the Others party received...`
- More relevant? Yes (top 3 all shifted to northern election records)

### What changed
The top hit shifted from an irrelevant budget chunk to election chunks about Upper West/Upper East. Scores increased substantially (0.555 to 0.761 at rank 1), showing that query expansion improved retrieval focus for ambiguous geographic wording.

Out-of-domain check (`What is the capital of France?`):
- `is_relevant`: True
- Top score: 0.525
- Observation: false positive remained, so thresholding alone did not reject this OOD query in this run.

---

## Log 3 — Prompt template comparison

**Date:** 2026-04-21
**Notebook:** `experiments/03_prompt_comparison.ipynb`
**Query tested:** "Who won the 2020 presidential election in Ghana?"

Template 1 (baseline):
`The provided context does not contain information about the overall winner of the 2020 Ghana presidential election. It only provides vote counts for specific candidates in certain regions.`
- Added information not in the context? No
- Tone: factual and concise

Template 2 (hallucination-controlled):
`I don't have enough information in the knowledge base to answer this question.`
- Refused to speculate? Yes
- Cited the context explicitly? Indirectly (explicit refusal due insufficient KB context)

Template 3 (with memory):
`The retrieved context is insufficient to determine the winner of the 2020 presidential election in Ghana. The provided information only includes vote counts for specific candidates in certain regions, but does not provide a comprehensive overview of the election results or the overall winner.`
- Did memory affect the answer? Slightly; response is more explanatory but same core conclusion
- More contextually coherent than Template 1? Yes (more complete explanation)

Context window management output:
- Chunks before: 5
- Chunks after: 5
- Total chars used: 605

### Which template is best and why
Template 2 is strictest for hallucination control. Template 3 gives the most chat-friendly explanation while still refusing unsupported claims, so Template 3 remains a practical default for multi-turn use.

---

## Log 4 — Adversarial tests

**Date:** 2026-04-21
**Notebook:** `experiments/04_adversarial_tests.ipynb`

### Query 1: "Who won?" (ambiguous)

RAG response:
`The context is insufficient to determine the winner of the election, as it only provides information about the votes received by Christian Kwabena Andrews of the GUM party in different regions. It does not provide information about the overall winner of the election or the votes received by other candidates.`
- Hallucination? No explicit fabricated winner
- Did it ask for clarification or admit ambiguity? Yes, it admitted insufficient context
- Top combined score: Not printed in notebook output (top chunk text shown only)

Pure LLM response (no retrieval):
`You didn't provide a question about a specific event or competition, so I'm not sure who "won" what. Could you please provide more context or information about the question you're asking? I'll do my best to provide a helpful answer.`
- Hallucination? No
- Did it invent a specific winner? No

RAG vs LLM: Both avoided hallucinating a winner. RAG attempted a domain-grounded explanation but retrieval relevance was weak for this ambiguous query.

---

### Query 2: "What did the president say about free healthcare in the 2025 budget?"

RAG response:
`According to the retrieved context, the president committed to deliver Free Primary Healthcare, and it is stated that "We are delivering!" (Source: Budget, Page: 136, xiii). Additionally, the president also committed to deliver the MahamaCares programme to finance the treatment of non-communicable diseases, and it is stated that "We are delivering!" (Source: Budget, Page: 136, xiv).`
- Confidence threshold triggered (score < 0.25)? No
- Top combined score: 0.6567
- Did it correctly refuse to fabricate? Not a refusal; it provided a source-grounded answer from retrieved budget context

Pure LLM response (no retrieval):
`I'm not aware of any information about a president's statement on free healthcare in the 2025 budget. As my knowledge cutoff is December 2023, I do not have real-time information or updates on events that may have occurred after that date. If you're looking for information on a specific president's statement or the 2025 budget, I recommend checking the latest news sources or official government websites for the most up-to-date information.`
- Did it hallucinate a presidential quote? No
- Did it cite a specific speech or document? No

RAG vs LLM: RAG produced a concrete sourced answer from the budget chunks; pure LLM gave a cautious generic refusal.

---

## Log 5 — Memory test

**Date:** 2026-04-21
**Test:** Multi-turn conversation to verify memory injection works

Turn 1:
- Query: "Who won the 2020 presidential election in Ghana?"
- Response: "The retrieved context is insufficient to determine the winner of the 2020 presidential election in Ghana. The provided information only includes vote counts for specific candidates in certain regions, but does not include overall national results or the winner of the election."

Turn 2:
- Query: "What about 2016?"
- Response: "I don't have enough relevant information in the knowledge base to answer this question." + low-confidence warning.
- Did it understand "2016" as referring to the election? Not reliably (this short follow-up triggered low-confidence fallback)
- Was the prior turn visible in the "Final prompt" expander? Yes

Turn 3 (clarified follow-up):
- Query: "What about 2016 election results in Ghana?"
- Response: The system returned regional 2016 election results for John Dramani Mahama (NDC) and Nana Akufo Addo (NPP), including vote counts/percentages for regions such as Northern, Central, Western, and Brong Ahafo.
- Did it understand the clarified follow-up? Yes

What I observed:
- The memory component helped only when the follow-up query was specific enough for retrieval. The short follow-up ("What about 2016?") failed with low confidence, but the clarified follow-up produced relevant 2016 regional results. This suggests memory alone is not sufficient when the retrieval signal is too weak.
- Conversation History (Q1/A1) was present in the final prompt for the follow-up query, confirming memory injection worked when retrieval was relevant.

---

## Summary

| Experiment | Key finding | Effect on system |
|---|---|---|
| Chunking | Election chunks were short/consistent; budget chunks preserved paragraph context | Improved retrieval structure for both datasets |
| Retrieval failure | Query expansion improved northern-region recall, but an OOD false positive remained (`capital of France` scored 0.525 and passed threshold) | Improved vague-query handling; OOD filtering still needs strengthening |
| Prompt comparison | All templates refused unsupported winner claim; Template 3 gave best chat-style explanation | Kept Template 3 as default for multi-turn UX |
| Adversarial Q1 | Both RAG and pure LLM avoided naming a winner, but retrieval relevance was weak | Shows need for stronger ambiguity handling |
| Adversarial Q2 | RAG produced source-grounded budget answer; pure LLM gave generic refusal | Demonstrated value of retrieval grounding |
| Memory | Ambiguous follow-up failed, but clarified follow-up returned coherent 2016 regional results | Multi-turn performance improves when follow-up queries are explicit |
