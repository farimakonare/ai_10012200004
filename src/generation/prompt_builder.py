"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004
Module: prompt_builder.py — Prompt templates and context window management.
"""

from typing import List, Dict, Any

CHARS_PER_TOKEN = 4
MAX_CONTEXT_TOKENS = 2000
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN


TEMPLATE_1_BASELINE = """\
Answer the following question using only the information in the context below.

Context:
{context}

Question: {query}
Answer:"""

TEMPLATE_2_HALLUCINATION_CONTROLLED = """\
You are a factual assistant specializing in Ghanaian politics, elections, and economic policy.
Answer ONLY using the provided context. Do NOT guess, invent, or rely on prior knowledge.
If the context does not contain sufficient information to answer, respond with:
"I don't have enough information in the knowledge base to answer this question."

Context:
{context}

Question: {query}
Answer:"""

TEMPLATE_3_WITH_MEMORY = """\
You are a factual assistant for Academic City University, specialized in Ghana's elections \
and economic policy.
Answer ONLY using the retrieved context below.
If the context is partial, still provide the best direct answer first, then state what is missing.
Only say you cannot determine the answer when none of the retrieved facts address the question.
Do not fabricate information.

{memory_block}Retrieved Context:
{context}

Current Question: {query}
Answer:"""


def _format_memory_block(memory_text: str) -> str:
    if not memory_text.strip():
        return ""
    return f"Conversation History:\n{memory_text}\n\n"


def manage_context_window(
    results: List[Dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> List[Dict[str, Any]]:
    """Keep top-scoring chunks that fit within the character budget."""
    selected = []
    total_chars = 0
    for r in results:
        chunk_len = len(r["chunk"]["text"])
        if total_chars + chunk_len > max_chars:
            break
        selected.append(r)
        total_chars += chunk_len
    return selected


def format_context(results: List[Dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        meta = r["chunk"]["metadata"]
        source_tag = f"[Source: {meta.get('source', 'unknown').title()}"
        if "year" in meta:
            source_tag += f", Year: {meta['year']}"
        if "region" in meta:
            source_tag += f", Region: {meta['region']}"
        if "page" in meta:
            source_tag += f", Page: {meta['page']}"
        source_tag += "]"
        lines.append(f"{i}. {source_tag}\n{r['chunk']['text']}")
    return "\n\n".join(lines)


def build_prompt(
    query: str,
    results: List[Dict[str, Any]],
    memory_text: str = "",
    template_id: int = 3,
) -> str:
    selected = manage_context_window(results)
    context = format_context(selected) if selected else "No relevant context retrieved."

    if template_id == 1:
        return TEMPLATE_1_BASELINE.format(context=context, query=query)
    elif template_id == 2:
        return TEMPLATE_2_HALLUCINATION_CONTROLLED.format(context=context, query=query)
    else:
        memory_block = _format_memory_block(memory_text)
        return TEMPLATE_3_WITH_MEMORY.format(
            memory_block=memory_block,
            context=context,
            query=query,
        )
