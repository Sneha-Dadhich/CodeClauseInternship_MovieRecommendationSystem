'''
- In machine learning, two primary methods of building recommendation engines are Content-based 
 and Collaborative filtering methods.
- When using the content-based filter method, the suggested products or items are based on what you 
 liked or purchased. This method feeds the machine learning model with historical data such as
 customer search history, purchase records, and items in their wishlists. The model finds other 
 products that share features similar to your past preferences.
- Imagine if all the recommendation systems just suggested things based on what you have seen. How 
 would you discover new genres and movies? That’s where the Collaborative filtering method comes in. 
 So what is it?
- Rather than finding similar content, the Collaborative filtering method finds other users and 
 customers similar to you and recommends their choices. The algorithm doesn’t consider the product
 features as in the case of content-based filtering.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from seaborn import load_dataset

# Importing the files
path = "C:\\Sneha\\Programs1\\Python\\Internship\\CodeClause\\MovieRecommendationSystem\\data\\file.tsv"
movie_path = "C:\\Sneha\\Programs1\\Python\\Internship\\CodeClause\\MovieRecommendationSystem\\data\\Movie_Id_Titles.csv"

# Get the data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
raw_data = pd.read_csv(path, sep='\t', names=column_names)
movie_titles = pd.read_csv(movie_path)

# Merging the data
data = pd.merge(raw_data, movie_titles, on='item_id')

def get_movie_recommendations(movie_title):
    
    # Calculate mean rating of all movies
    mean_ratings = data.groupby('title')['rating'].mean().sort_values(ascending=False)

    # Creating dataframe with 'rating' count values
    ratings = pd.DataFrame(data.groupby('title')['rating'].mean())
    ratings['num_of_ratings'] = pd.DataFrame(data.groupby('title')['rating'].count())

    # Sorting values according to the 'num_of_rating' column
    moviemat = data.pivot_table(index='user_id', columns='title', values='rating')

    # Analysing correlation with similar movies
    user_ratings = moviemat[movie_title]
    similar_to_movie = moviemat.corrwith(user_ratings)

    # Create DataFrame for correlation
    corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)

    # Join with ratings
    corr_movie = corr_movie.join(ratings['num_of_ratings'])

    # Filter movies with minimum number of ratings
    min_ratings = 100
    recommended_movies = corr_movie[corr_movie['num_of_ratings'] > min_ratings].sort_values('Correlation', ascending=False)

    # Add average rating column
    recommended_movies = recommended_movies.join(mean_ratings, on='title')
    recommended_movies.rename(columns={'rating': 'average_rating'}, inplace=True)
    return recommended_movies

# Example usage:
movie_title = "Star Wars (1977)"  # Change this to any movie title from your dataset
recommendations = get_movie_recommendations(movie_title)

def main():
    st.set_page_config(layout="wide")

    st.title('Movie Recommendation System')
    st.sidebar.title('Options')

    # Load your dataset here
    # Load your dataset using appropriate function

    movie_title = st.sidebar.selectbox('Select a movie:', data['title'].unique())

    recommendations = get_movie_recommendations(movie_title)

    st.subheader('Movie Recommendations for ' + movie_title)
    st.write(recommendations.head(10))

    st.subheader('Data Sample')
    st.write(data.head())

if __name__ == '__main__':
    main()