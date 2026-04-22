"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004

Script: Download raw datasets from official sources.
Run once before building the FAISS index.
"""

import os
import urllib.request

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

DATASETS = {
    "Ghana_Election_Result.csv": (
        "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
    ),
    "2025-Budget-Statement.pdf": (
        "https://mofep.gov.gh/sites/default/files/budget-statements/"
        "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    ),
}


def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    for filename, url in DATASETS.items():
        dest = os.path.join(RAW_DIR, filename)
        if os.path.exists(dest):
            print(f"  [skip] {filename} already exists")
            continue
        print(f"  [download] {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            size_kb = os.path.getsize(dest) / 1024
            print(f"  [ok] {filename} ({size_kb:.0f} KB)")
        except Exception as exc:
            print(f"  [error] {filename}: {exc}")


if __name__ == "__main__":
    print("Downloading datasets...")
    download()
    print("Done.")
