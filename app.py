"""
app.py  –  Movie Recommendation System
──────────────────────────────────────
Run locally:
    streamlit run app.py

Deploy on Streamlit Cloud:
    Push this file + movie_recommender.pkl + requirements.txt to GitHub,
    then connect the repo at https://share.streamlit.io
"""

import pickle
import numpy as np
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str = "movie_recommender.pkl"):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["movie_data"], bundle["similarity"]


try:
    movie_data, similarity = load_model()
except FileNotFoundError:
    st.error(
        "**movie_recommender.pkl not found.**\n\n"
        "Run `python build_model.py --data movie_dataset.csv` first, "
        "then place the generated `.pkl` next to `app.py`."
    )
    st.stop()

movie_titles = movie_data["title"].tolist()


# ── Core recommendation logic ─────────────────────────────────────────────────
def recommend(title: str, top_n: int = 5):
    """Return top-N similar movie titles for the given title."""
    matches = movie_data[movie_data["title"].str.lower() == title.lower()]
    if matches.empty:
        return None, []

    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Skip the movie itself (index 0 after sorting is always itself)
    top = [movie_data.iloc[i[0]]["title"] for i in scores[1: top_n + 1]]
    return title, top


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.markdown(
    "Select a movie you love and we'll suggest **5 similar movies** "
    "based on genres, keywords, cast, and director."
)

st.divider()

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
        _, recommendations = recommend(selected_movie, top_n=num_recs)

        if not recommendations:
            st.error(f"Could not find **{selected_movie}** in the dataset.")
        else:
            st.success(f"Because you liked **{selected_movie}**, you might also enjoy:")
            st.divider()

            for i, movie in enumerate(recommendations, start=1):
                st.markdown(f"**{i}.** {movie}")

st.divider()
st.caption(
    f"Dataset: {len(movie_titles):,} movies  •  "
    "Model: Content-based filtering (TF bag-of-words + cosine similarity)"
)
