"""
app.py  –  Movie Recommendation System
──────────────────────────────────────
No .pkl needed! The model is built from movie_dataset.csv on first run
and cached automatically by Streamlit.

Run locally:    streamlit run app.py
Deploy:         Push app.py + movie_dataset.csv + requirements.txt to GitHub
                then connect at https://share.streamlit.io
"""

import ast
import os

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

# ── Model building helpers ────────────────────────────────────────────────────

def extract_names(obj, top_n=None):
    try:
        items = ast.literal_eval(obj)
        names = [i["name"].replace(" ", "_") for i in items]
        return names[:top_n] if top_n else names
    except Exception:
        return []


def build_tags(row):
    overview  = [w.lower() for w in str(row["overview"]).split()] if pd.notna(row["overview"]) else []
    genres    = extract_names(row["genres"])
    keywords  = extract_names(row["keywords"])
    cast      = extract_names(row["cast"], top_n=3)
    director  = [row["director"].replace(" ", "_")] if isinstance(row["director"], str) else []
    return " ".join(overview + genres + keywords + cast + director).lower()


# ── Load / build model ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(csv_path: str = "movie_dataset.csv"):
    """
    Build the recommendation model from the CSV.
    Streamlit caches this so it only runs once per session.
    """
    if not os.path.exists(csv_path):
        return None, None, f"Dataset file '{csv_path}' not found."

    df = pd.read_csv(csv_path)
    required = {"id", "title", "overview", "genres", "keywords", "cast", "director"}
    missing = required - set(df.columns)
    if missing:
        return None, None, f"Dataset is missing columns: {missing}"

    df = df[list(required)].dropna().reset_index(drop=True)
    df["tags"] = df.apply(build_tags, axis=1)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    movie_data = df[["id", "title"]].copy()
    return movie_data, similarity, None


# ── Core recommendation logic ─────────────────────────────────────────────────

def recommend(title: str, movie_data, similarity, top_n: int = 5):
    matches = movie_data[movie_data["title"].str.lower() == title.lower()]
    if matches.empty:
        return []
    idx = matches.index[0]
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    return [movie_data.iloc[i[0]]["title"] for i in scores[1: top_n + 1]]


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🎬 Movie Recommendation System")
st.markdown(
    "Select a movie you love and we'll suggest similar movies "
    "based on genres, keywords, cast, and director."
)
st.divider()

with st.spinner("⚙️ Loading model… (this takes ~15 seconds on first run)"):
    movie_data, similarity, error = load_model()

if error:
    st.error(f"**Error:** {error}")
    st.stop()

movie_titles = movie_data["title"].tolist()

selected_movie = st.selectbox(
    "🔍 Choose a movie",
    options=sorted(movie_titles),
    index=None,
    placeholder="Start typing a movie name …",
)

num_recs = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
    if not selected_movie:
        st.warning("Please select a movie first.")
    else:
        results = recommend(selected_movie, movie_data, similarity, top_n=num_recs)
        if not results:
            st.error(f"Could not find **{selected_movie}** in the dataset.")
        else:
            st.success(f"Because you liked **{selected_movie}**, you might also enjoy:")
            st.divider()
            for i, movie in enumerate(results, start=1):
                st.markdown(f"**{i}.** {movie}")

st.divider()
st.caption(
    f"Dataset: {len(movie_titles):,} movies  •  "
    "Model: Content-based filtering (TF bag-of-words + cosine similarity)"
)
