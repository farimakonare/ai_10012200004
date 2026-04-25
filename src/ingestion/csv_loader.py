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
    df[str_cols] = df[str_cols].apply(
        lambda col: col.str.replace("\xa0", " ", regex=False).str.strip()
    )
    return df


def _safe_int(value: Any) -> int:
    txt = re.sub(r"[^\d]", "", str(value))
    return int(txt) if txt else 0


def _safe_year(value: Any) -> Any:
    try:
        return int(float(value))
    except Exception:
        return value


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
    new_region = _get(["new_region", "region"])
    old_region = _get(["old_region"])

    if new_region and old_region and new_region != old_region:
        region_text = f"{new_region} (formerly part of {old_region})"
    else:
        region_text = new_region or old_region or "Ghana"

    if candidate and year and party:
        return (
            f"In the {year} Ghana presidential election, {candidate} of the {party} party"
            f" received {votes} votes ({votes_pct}) in {region_text}."
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

    year_col = next((c for c in cols if c == "year"), None)
    new_region_col = next((c for c in cols if c == "new_region"), None)
    old_region_col = next((c for c in cols if c == "old_region"), None)
    party_col = next((c for c in cols if c == "party"), None)
    candidate_col = next((c for c in cols if c == "candidate"), None)
    votes_col = next((c for c in cols if c == "votes"), None)

    for idx, row in df.iterrows():
        text = _row_to_text(row, cols)
        if not text.strip():
            continue
        meta: Dict[str, Any] = {
            "source": "election",
            "doc_type": "row_result",
            "row_id": int(idx),
        }
        if year_col and not pd.isna(row[year_col]):
            meta["year"] = _safe_year(row[year_col])
        if new_region_col and not pd.isna(row[new_region_col]):
            meta["region"] = str(row[new_region_col]).strip()
        elif old_region_col and not pd.isna(row[old_region_col]):
            meta["region"] = str(row[old_region_col]).strip()
        if old_region_col and not pd.isna(row[old_region_col]):
            meta["old_region"] = str(row[old_region_col]).strip()
        if party_col and not pd.isna(row[party_col]):
            meta["party"] = str(row[party_col]).strip()
        if candidate_col and not pd.isna(row[candidate_col]):
            meta["candidate"] = str(row[candidate_col]).strip()
        docs.append({"text": text, "metadata": meta})

    # Add one national summary doc per year so winner-style queries retrieve direct facts.
    if all([year_col, candidate_col, party_col, votes_col]):
        summary = (
            df.assign(_votes_int=df[votes_col].map(_safe_int))
            .groupby([year_col, candidate_col, party_col], dropna=False, as_index=False)["_votes_int"]
            .sum()
            .sort_values([year_col, "_votes_int"], ascending=[True, False])
        )
        for year, grp in summary.groupby(year_col):
            top = grp.head(2).to_dict("records")
            if not top:
                continue
            winner = top[0]
            runner = top[1] if len(top) > 1 else None
            year_val = _safe_year(year)
            text = (
                f"National summary for the {year_val} Ghana presidential election: "
                f"winner was {winner[candidate_col]} ({winner[party_col]}) with "
                f"{winner['_votes_int']:,} votes."
            )
            if runner:
                text += (
                    f" Runner-up was {runner[candidate_col]} ({runner[party_col]}) with "
                    f"{runner['_votes_int']:,} votes."
                )
            docs.append({
                "text": text,
                "metadata": {
                    "source": "election",
                    "doc_type": "national_summary",
                    "year": year_val,
                    "region": "National",
                    "party": str(winner[party_col]).strip(),
                    "candidate": str(winner[candidate_col]).strip(),
                },
            })

    return docs


def load_election_documents(path: str = RAW_CSV) -> List[Dict[str, Any]]:
    df = load_and_clean(path)
    return to_documents(df)


if __name__ == "__main__":
    docs = load_election_documents()
    print(f"Loaded {len(docs)} election documents")
    if docs:
        print("Sample:", docs[0])
