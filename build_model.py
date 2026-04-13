"""
build_model.py
──────────────
Run this once to generate movie_recommender.pkl
Usage:  python build_model.py --data movie_dataset.csv
"""

import argparse
import ast
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_names(obj, top_n=None):
    """Parse a JSON-ish list-of-dicts string and return name tokens."""
    try:
        items = ast.literal_eval(obj)
        names = [i["name"].replace(" ", "_") for i in items]
        return names[:top_n] if top_n else names
    except Exception:
        return []


def build_tags(row):
    overview   = [w.lower() for w in str(row["overview"]).split()] if pd.notna(row["overview"]) else []
    genres     = extract_names(row["genres"])
    keywords   = extract_names(row["keywords"])
    cast       = extract_names(row["cast"], top_n=3)
    director   = [row["director"].replace(" ", "_")] if isinstance(row["director"], str) else []
    return " ".join(overview + genres + keywords + cast + director).lower()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(csv_path: str, out_path: str = "movie_recommender.pkl"):
    print(f"📂  Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"id", "title", "overview", "genres", "keywords", "cast", "director"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df = df[list(required)].dropna().reset_index(drop=True)
    print(f"    {len(df)} movies after dropping nulls")

    print("🏷️   Building feature tags …")
    df["tags"] = df.apply(build_tags, axis=1)

    print("🔢  Vectorising (CountVectorizer, max_features=5000) …")
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()

    print("📐  Computing cosine similarity …")
    similarity = cosine_similarity(vectors)

    bundle = {
        "movie_data": df[["id", "title"]].copy(),
        "similarity": similarity,
    }

    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"✅  Saved → {out_path}  ({similarity.shape[0]} movies)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="movie_dataset.csv", help="Path to CSV")
    parser.add_argument("--out",  default="movie_recommender.pkl", help="Output .pkl path")
    args = parser.parse_args()
    main(args.data, args.out)
