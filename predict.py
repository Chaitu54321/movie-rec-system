import joblib
import pandas as pd
import streamlit as st

# Load trained model + data
@st.cache_resource
def load_assets():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("movie_model.pkl")
    df = pd.read_csv("clean_movies.csv")
    return vectorizer, model, df

vectorizer, model, df = load_assets()

# Streamlit UI
st.title("🎥 Movie Recommendation System (from saved model)")

# Dropdown for genres
all_genres = sorted({g for sublist in df["genres"].str.split('|') for g in sublist})
selected_genres = st.multiselect("Select your favorite genres:", all_genres)

# Year slider
min_year, max_year = int(df["year"].min()), int(df["year"].max())
year_range = st.slider("Select year range:", min_year, max_year, (1990, 2020))

if st.button("Recommend Movies"):
    if not selected_genres:
        st.warning("Please select at least one genre.")
    else:
        user_input = " ".join(selected_genres)
        user_vec = vectorizer.transform([user_input])

        distances, indices = model.kneighbors(user_vec, n_neighbors=20)
        recs = df.iloc[indices[0]]
        recs = recs[(recs["year"] >= year_range[0]) & (recs["year"] <= year_range[1])]

        if recs.empty:
            st.error("No movies found for given criteria.")
        else:
            st.subheader("🎬 Recommended Movies:")
            for _, row in recs.iterrows():
                st.write(f"**{row['title']}** — {row['genres']}")
