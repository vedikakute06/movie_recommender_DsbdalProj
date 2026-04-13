import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD & PROCESS DATA ---------------- #

@st.cache_resource
def load_data():
    df = pd.read_csv('movie_dataset.csv')

    features = ['keywords', 'cast', 'genres', 'director', 'overview']

    for feature in features:
        df[feature] = df[feature].fillna('')

    def combine_features(row):
        return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director'] + " " + row['overview']

    df["combined_features"] = df.apply(combine_features, axis=1)

    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(df["combined_features"])

    return df, vectors

df, vectors = load_data()

# ---------------- RECOMMEND FUNCTION ---------------- #

def recommend(movie):
    index = df[df['title'] == movie].index[0]

    similarity = cosine_similarity(vectors[index], vectors).flatten()

    movie_list = sorted(list(enumerate(similarity)), key=lambda x: x[1], reverse=True)[1:6]

    return [df.iloc[i[0]].title for i in movie_list]

# ---------------- STREAMLIT UI ---------------- #

st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Search or select a movie",
    df['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)
