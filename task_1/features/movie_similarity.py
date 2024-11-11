import numpy as np
import pandas as pd

from task_1.features import cosine_similarity, jaccard_similarity
from task_1.movie.tmdb.client import Client


def load_data(file_path):
    """Load the ratings data from a CSV file."""
    return pd.read_csv(file_path, sep=';', header=0)

def create_ratings_matrix(data):
    """Create a user-movie ratings matrix from the data."""
    return data.pivot_table(index='UserID', columns='MovieID', values='Rating', fill_value=0)


def create_movie_mapping(file_path: str) -> dict:
    """Create a mapping of movie titles to TMDBID from the given CSV file."""
    # Load the movie data from the provided CSV file
    movie_data = pd.read_csv(file_path, sep=';', header=0)

    # Create a dictionary that maps movie titles to their TMDBID
    movie_mapping = {row['Title']: row['TMDBID'] for _, row in movie_data.iterrows()}

    return movie_mapping


def calculate_movie_similarity(movie1, movie2, ratings_matrix=None, movie_mapping=None):
    """Calculate similarity between two movies based on features, genres, cast, and ratings."""

    # Calculate numerical feature similarity using cosine similarity
    numerical_features_1 = movie1.to_feature_vector()
    numerical_features_2 = movie2.to_feature_vector()
    numerical_similarity = cosine_similarity(numerical_features_1, numerical_features_2)

    # Calculate Jaccard similarity for genres
    genres_similarity = jaccard_similarity(set(movie1.genres), set(movie2.genres))

    # Calculate Jaccard similarity for cast
    cast_similarity = jaccard_similarity(set(movie1.cast), set(movie2.cast))

    # Calculate rating similarity if ratings_matrix is provided
    if ratings_matrix is not None and movie_mapping is not None:
        # Map movie1 and movie2 TMDBIDs to their corresponding MovieID in the ratings_matrix
        movie1_movieid = movie_mapping.get(movie1.title, None)
        movie2_movieid = movie_mapping.get(movie2.title, None)

        # Ensure both movie IDs exist in the ratings matrix
        if movie1_movieid is not None and movie2_movieid is not None:
            # Get the rating vectors for both movies based on MovieID (TMDBID)
            movie_1_ratings = ratings_matrix[movie1_movieid].values
            movie_2_ratings = ratings_matrix[movie2_movieid].values

            # Calculate cosine similarity for ratings
            ratings_similarity = cosine_similarity(movie_1_ratings, movie_2_ratings)
        else:
            # If ratings for one or both movies are missing, assign a similarity of 0
            ratings_similarity = 0

    else:
        # If no ratings matrix or mapping is provided, default to 0 for rating similarity
        ratings_similarity = 0

        # Combine all similarities with adjusted weights
    total_similarity = (6 / 9) * numerical_similarity + (1 / 9) * genres_similarity + (1 / 9) * cast_similarity + (
                1 / 9) * ratings_similarity
    return total_similarity






