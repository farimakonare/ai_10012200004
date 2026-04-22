"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004
Module: csv_loader.py — Load and clean the Ghana Election Results CSV.
"""

import os
import re
import pandas as pd
from typing import List, Dict, Any


RAW_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "Ghana_Election_Result.csv")


def _normalize_col(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def load_and_clean(path: str = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df.columns = [_normalize_col(c) for c in df.columns]

    critical = [c for c in df.columns if any(k in c for k in ("year", "party", "vote", "region", "seat"))]
    df = df.dropna(subset=critical if critical else df.columns[:3])
    df = df.reset_index(drop=True)

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    return df


def _row_to_text(row: pd.Series, columns: List[str]) -> str:
    """Convert a DataFrame row into a natural-language sentence for embedding."""
    def _get(keys):
        for k in keys:
            for col in columns:
                if col.lower().replace(" ", "_") == k or col.lower() == k:
                    val = row[col]
                    if not pd.isna(val) and str(val).strip():
                        return str(val).strip()
        return None

    year      = _get(["year"])
    candidate = _get(["candidate"])
    party     = _get(["party"])
    votes     = _get(["votes"])
    votes_pct = _get(["votes(%)"])
    region    = _get(["old_region", "new_region", "region"])

    if candidate and year and party:
        return (
            f"In the {year} Ghana presidential election, {candidate} of the {party} party"
            f" received {votes} votes ({votes_pct}) in {region}."
        )

    # Fallback: generic key-value format
    parts = []
    for col in columns:
        val = row[col]
        if pd.isna(val) or str(val).strip() == "":
            continue
        parts.append(f"{col.replace('_', ' ').title()}: {val}")
    return " | ".join(parts)


def to_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    docs = []
    cols = list(df.columns)
    for idx, row in df.iterrows():
        text = _row_to_text(row, cols)
        if not text.strip():
            continue
        meta: Dict[str, Any] = {"source": "election", "row_id": int(idx)}
        for key in ("year", "region", "constituency", "party"):
            match = next((c for c in cols if key in c), None)
            if match and not pd.isna(row[match]):
                meta[key] = row[match]
        docs.append({"text": text, "metadata": meta})
    return docs


def load_election_documents(path: str = RAW_CSV) -> List[Dict[str, Any]]:
    df = load_and_clean(path)
    return to_documents(df)


if __name__ == "__main__":
    docs = load_election_documents()
    print(f"Loaded {len(docs)} election documents")
    if docs:
        print("Sample:", docs[0])
