"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004
Module: llm_client.py — Groq API client with exponential-backoff retry.
"""

import os
import time

from groq import Groq, RateLimitError, APIError


MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
TEMPERATURE = 0.1   # low temperature for factual, deterministic responses
MAX_RETRIES = 3


def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Set it as an environment variable or "
            "add it to .streamlit/secrets.toml."
        )
    return Groq(api_key=api_key)


def generate(
    prompt: str,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    client = _get_client()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            if attempt == MAX_RETRIES:
                return "The system is temporarily busy. Please try again in a few seconds."
            time.sleep(2 ** attempt)  # 2, 4, 8 seconds

        except APIError as exc:
            return f"LLM API error: {exc}"

        except Exception as exc:
            return f"Unexpected error during generation: {exc}"

    return "Generation failed after maximum retries."


def generate_no_context(query: str) -> str:
    """Pure LLM mode — no retrieved context. Used in adversarial comparison tests."""
    prompt = f"Answer the following question based on your general knowledge:\n\n{query}\n\nAnswer:"
    return generate(prompt)
