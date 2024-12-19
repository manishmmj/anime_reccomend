import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO

# GitHub URLs for the dataset and similarity matrix
DATA_URL = "C://Users//Manish//Downloads//DATA SCIENCE ASSIGNMENT EXCELR SOLUTION//anime_similarity.pkl"
MODEL_URL = "https://github.com/manishmmj/anime_reccomend/blob/main/anime_data.pkl"

# Load anime dataset
anime_df = pd.read_csv(DATA_URL)

# Load the precomputed cosine similarity matrix from GitHub
cosine_sim = pickle.load(BytesIO(requests.get(MODEL_URL).content))

# Streamlit App Title
st.title("Anime Recommendation System")

# Instructions for the user
st.write("""
    ### Welcome to the Anime Recommendation System!
    Please select your favorite anime from the dropdown below, and we will recommend similar anime titles based on your choice.
""")

# User Input for Anime Preferences
anime_list = anime_df['name'].tolist()
selected_anime = st.selectbox('Select an Anime:', anime_list)

num_recommendations = st.slider('Number of Recommendations:', min_value=1, max_value=10, value=5)

# Function to recommend anime based on cosine similarity
def recommend_anime(name, n_recommendations=5):
    # Find the index of the selected anime
    indices = pd.Series(anime_df.index, index=anime_df['name']).drop_duplicates()
    idx = indices[name]
    
    # Get similarity scores and sort them
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top n recommendations (excluding the selected anime)
    sim_scores = sim_scores[1:n_recommendations + 1]
    anime_indices = [i[0] for i in sim_scores]
    
    # Return recommended anime
    return anime_df[['name', 'genre', 'type', 'episodes', 'rating']].iloc[anime_indices]

# Display recommendations when button is clicked
if st.button('Recommend Anime'):
    recommendations = recommend_anime(selected_anime, num_recommendations)
    st.write(f"### Recommended Anime Similar to '{selected_anime}':")
    st.dataframe(recommendations)
