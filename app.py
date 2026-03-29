import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("tmdb_5000_movies.csv")

df = load_data()

# -----------------------------
# PREPROCESS
# -----------------------------
title_col = 'original_title' if 'original_title' in df.columns else 'title'
df = df[[title_col, 'overview']].dropna()

# -----------------------------
# UI
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.write("Find similar movies using NLP and cosine similarity")

search = st.text_input("Search movie")

movie_list = df[title_col].values

if search:
    filtered = [m for m in movie_list if search.lower() in m.lower()]
else:
    filtered = movie_list[:200]

selected_movie = st.selectbox("Select Movie", filtered)

# -----------------------------
# RECOMMENDATION
# -----------------------------
if st.button("🚀 Recommend"):

    with st.spinner("Generating recommendations..."):

        # NLP
        cv = CountVectorizer(stop_words='english')
        vectors = cv.fit_transform(df['overview']).toarray()

        # Similarity
        similarity = cosine_similarity(vectors)

        # Recommend
        index = df[df[title_col] == selected_movie].index[0]
        distances = similarity[index]

        movies = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:6]

        results = [df.iloc[i[0]][title_col] for i in movies]

        st.subheader("🎯 Recommended Movies")

        for movie in results:
            st.write("👉", movie)

# -----------------------------
# EXPLANATION
# -----------------------------
st.markdown("## 🧠 How it works")
st.write("""
This system uses NLP (CountVectorizer) to convert text into numerical form.
Then cosine similarity is used to find movies with similar descriptions.
""")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.write("Developed by Shubham")

# To run this project
# python -m streamlit run app.py   