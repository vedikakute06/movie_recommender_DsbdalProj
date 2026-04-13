# 🎬 Movie Recommendation System

A content-based movie recommendation engine built with **scikit-learn** and deployed via **Streamlit**. Type in any movie from a 4800+ title dataset and instantly get 5 (or more) similar recommendations — powered by cosine similarity over movie metadata.

---

## 📸 App Preview

```
┌─────────────────────────────────────────────┐
│  🎬 Movie Recommendation System             │
│                                             │
│  🔍 Choose a movie                          │
│  [ Avatar                              ▼ ]  │
│                                             │
│  Number of recommendations:  ●──── 5        │
│                                             │
│       [ 🎯 Get Recommendations ]            │
│                                             │
│  ✅ Because you liked Avatar, you might     │
│     also enjoy:                             │
│                                             │
│   1. Guardians of the Galaxy                │
│   2. Star Trek Into Darkness                │
│   3. Aliens                                 │
│   4. The Avengers                           │
│   5. Interstellar                           │
└─────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
movie-recommender/
├── app.py                   # Streamlit web application
├── build_model.py           # Model training script → outputs .pkl
├── movie_recommender.pkl    # Pre-trained model (ready to use)
├── movie_dataset.csv        # Source dataset (4800+ movies)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## ⚙️ How It Works

The system uses **content-based filtering** — it recommends movies similar to the one you pick based on shared content attributes, not user ratings.

### Pipeline

```
Raw CSV
  │
  ▼
Extract features:
  overview words + genres + keywords + top-3 cast + director
  │
  ▼
CountVectorizer  (bag-of-words, 5000 features, stop-words removed)
  │
  ▼
Cosine Similarity Matrix  (4375 × 4375)
  │
  ▼
Pickle (.pkl) → loaded by Streamlit at runtime
```

### Feature Tags (per movie)
| Feature | Example tokens |
|---|---|
| Overview | `paraplegic`, `marine`, `planet` |
| Genres | `Action`, `Science_Fiction` |
| Keywords | `alien`, `space_travel` |
| Cast (top 3) | `Sam_Worthington`, `Zoe_Saldana` |
| Director | `James_Cameron` |

All tokens are combined into a single string, lowercased, and vectorised. Similarity is then the cosine angle between any two movie vectors.

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender

pip install -r requirements.txt
```

### 2. (Optional) Rebuild the model

Only needed if you swap in a new dataset:

```bash
python build_model.py --data movie_dataset.csv
# Outputs: movie_recommender.pkl
```

### 3. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push all files to a **public GitHub repo**
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** → select your repo
4. Set **Main file path** → `app.py`
5. Click **Deploy**

Your app will be live at `https://YOUR_APP.streamlit.app` within ~2 minutes.

> **Note:** `movie_recommender.pkl` is ~50MB — well within GitHub's 100MB file limit. If you ever exceed it, enable [Git LFS](https://git-lfs.github.com/).

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.32 | Web UI framework |
| `scikit-learn` | ≥ 1.4 | CountVectorizer + cosine similarity |
| `pandas` | ≥ 2.1 | Data loading & manipulation |
| `numpy` | ≥ 1.26 | Matrix operations |

---

## 📊 Dataset

The model was trained on **4,803 movies** from a TMDB-style dataset with the following columns used:

`title` · `overview` · `genres` · `keywords` · `cast` · `crew` · `director`

After dropping rows with missing values, **4,375 movies** are available in the recommender.

---

## 🔧 Customisation

| What to change | Where |
|---|---|
| Number of features | `CountVectorizer(max_features=5000)` in `build_model.py` |
| How many cast members to include | `extract_names(row["cast"], top_n=3)` in `build_model.py` |
| Default recommendations shown | `value=5` in `st.slider(...)` in `app.py` |
| Max recommendations allowed | `max_value=10` in `st.slider(...)` in `app.py` |

---

## 📄 License

MIT — free to use, modify, and distribute.
