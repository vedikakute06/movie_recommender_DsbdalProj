import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 60%, #16213e 100%);
    padding: 3rem 2.5rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.07);
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    color: #f5e6c8;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}
.hero p {
    color: #9a9ab0;
    font-size: 1.05rem;
    margin: 0;
}
.hero .accent {
    color: #e8a045;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9a9ab0;
    margin-bottom: 0.5rem;
}

/* ── Movie cards ── */
.movie-card {
    background: #ffffff08;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: all 0.2s ease;
}
.movie-card:hover {
    background: #ffffff12;
    border-color: rgba(232, 160, 69, 0.3);
    transform: translateX(4px);
}
.movie-rank {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: #e8a04540;
    min-width: 2rem;
    text-align: center;
    line-height: 1;
}
.movie-title {
    font-size: 1rem;
    font-weight: 500;
    color: #f0ece4;
    margin: 0;
}
.movie-meta {
    font-size: 0.8rem;
    color: #6b6b80;
    margin: 0.15rem 0 0;
}

/* ── Info / stat cards ── */
.stat-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.stat-box {
    flex: 1;
    background: #ffffff06;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #e8a045;
    line-height: 1.1;
}
.stat-label {
    font-size: 0.72rem;
    color: #6b6b80;
    margin-top: 0.2rem;
    letter-spacing: 0.05em;
}

/* ── Pipeline steps ── */
.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.pipeline-step:last-child { border-bottom: none; }
.step-num {
    background: #e8a045;
    color: #0d0d0d;
    border-radius: 50%;
    width: 22px; height: 22px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700;
    flex-shrink: 0; margin-top: 2px;
}
.step-text strong { color: #f0ece4; font-size: 0.9rem; display: block; margin-bottom: 0.15rem; }
.step-text span { color: #6b6b80; font-size: 0.8rem; }

/* ── Selectbox tweak ── */
div[data-baseweb="select"] > div {
    background-color: #ffffff0a !important;
    border-color: rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #e8a045, #c47a1e);
    color: #0d0d0d;
    font-weight: 600;
    font-size: 0.95rem;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }

/* ── Tag pills ── */
.tag {
    display: inline-block;
    background: rgba(232,160,69,0.12);
    color: #e8a045 !important;
    border: 1px solid rgba(232,160,69,0.25);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 3px 3px 3px 0;
}

/* ── Dark global backdrop ── */
.stApp { background: #0a0a14; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_data():
    df = pd.read_csv('movie_dataset.csv')
    features = ['keywords', 'cast', 'genres', 'director', 'overview']
    for f in features:
        df[f] = df[f].fillna('')
    df["combined_features"] = (
        df['keywords'] + " " + df['cast'] + " " +
        df['genres'] + " " + df['director'] + " " + df['overview']
    )
    vectorizer = TfidfVectorizer(stop_words='english', max_features=15_000)
    vectors = vectorizer.fit_transform(df["combined_features"])
    return df, vectors, vectorizer

df, vectors, vectorizer = load_data()

# ─────────────────────────────────────────────
#  RECOMMEND
# ─────────────────────────────────────────────
def recommend(movie, n=5):
    idx = df[df['title'] == movie].index[0]
    sim = cosine_similarity(vectors[idx], vectors).flatten()
    ranked = sorted(enumerate(sim), key=lambda x: x[1], reverse=True)[1:n+1]
    results = []
    for i, score in ranked:
        row = df.iloc[i]
        results.append({
            "title": row['title'],
            "score": round(score * 100, 1),
            "genres": row.get('genres', ''),
            "director": row.get('director', ''),
        })
    return results

# ─────────────────────────────────────────────
#  SIDEBAR — Model Info
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.markdown("---")

    st.markdown("### How it works")
    steps = [
        ("Feature Fusion", "Keywords, cast, genres, director & overview merged into one text blob."),
        ("TF-IDF Vectorisation", "Each movie becomes a high-dim sparse vector. Rare but meaningful terms get boosted weight."),
        ("Cosine Similarity", "Measures the angle between vectors — closer angle = more similar taste profile."),
        ("Top-K Ranking", "The 5 nearest neighbours (excluding the query) are returned as recommendations."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="pipeline-step">
          <div class="step-num">{i}</div>
          <div class="step-text"><strong>{title}</strong><span>{desc}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dataset stats")
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-box"><div class="stat-value">{len(df):,}</div><div class="stat-label">Movies</div></div>
      <div class="stat-box"><div class="stat-value">{vectors.shape[1]:,}</div><div class="stat-label">TF-IDF features</div></div>
    </div>""", unsafe_allow_html=True)

    if 'genres' in df.columns:
        all_genres = [g.strip() for gs in df['genres'].dropna() for g in gs.split()]
        top_genres = pd.Series(all_genres).value_counts().head(8).index.tolist()
        tags = "".join(f'<span class="tag">{g}</span>' for g in top_genres)
        st.markdown("**Top genres**")
        st.markdown(tags, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Tech stack")
    st.markdown("""
    <span class="tag">scikit-learn</span>
    <span class="tag">TF-IDF</span>
    <span class="tag">cosine similarity</span>
    <span class="tag">pandas</span>
    <span class="tag">streamlit</span>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎬 CineMatch</h1>
  <p>Content-based movie recommendations powered by <span class="accent">TF-IDF</span> &amp; <span class="accent">cosine similarity</span></p>
</div>
""", unsafe_allow_html=True)

col_search, col_btn = st.columns([4, 1], gap="small")
with col_search:
    st.markdown('<div class="section-label">Pick a movie you love</div>', unsafe_allow_html=True)
    selected_movie = st.selectbox(
        label="",
        options=sorted(df['title'].dropna().unique()),
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown('<div style="margin-top:1.6rem"></div>', unsafe_allow_html=True)
    run = st.button("Find similar →")

# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
if run and selected_movie:
    recs = recommend(selected_movie)

    # Show selected movie info strip
    sel = df[df['title'] == selected_movie].iloc[0]
    with st.container():
        st.markdown(f"""
        <div class="movie-card" style="background:#e8a04510; border-color:rgba(232,160,69,0.3); margin-bottom:1.5rem">
          <div class="movie-rank">★</div>
          <div>
            <p class="movie-title" style="color:#e8a045; font-size:1.1rem">{sel['title']}</p>
            <p class="movie-meta">Dir. {sel.get('director','—')} &nbsp;·&nbsp; {sel.get('genres','')}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Top 5 recommendations</div>', unsafe_allow_html=True)
    for rank, rec in enumerate(recs, 1):
        genres_display = rec['genres'][:50] + "…" if len(rec['genres']) > 50 else rec['genres']
        st.markdown(f"""
        <div class="movie-card">
          <div class="movie-rank">{rank}</div>
          <div style="flex:1">
            <p class="movie-title">{rec['title']}</p>
            <p class="movie-meta">Dir. {rec['director'] or '—'} &nbsp;·&nbsp; {genres_display}</p>
          </div>
          <div style="text-align:right">
            <div style="font-size:1.1rem;font-weight:600;color:#e8a045">{rec['score']}%</div>
            <div style="font-size:0.72rem;color:#6b6b80">similarity</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

elif not run:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #4a4a60;">
      <div style="font-size:3rem; margin-bottom:1rem">🍿</div>
      <p style="font-size:1rem">Select a movie above and hit <strong style="color:#e8a045">Find similar →</strong></p>
    </div>
    """, unsafe_allow_html=True)
