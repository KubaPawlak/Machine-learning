import numpy as np
import pandas as pd

from task_1.features import cosine_similarity, jaccard_similarity
from task_1.tmdb.client import Client


def load_data(file_path):
    """Load the ratings data from a CSV file."""
    return pd.read_csv(file_path, sep=';', header=0)

def create_ratings_matrix(data):
    """Create a user-movie ratings matrix from the data."""
    return data.pivot_table(index='UserID', columns='MovieID', values='Rating', fill_value=0)

def calculate_movie_similarity(movie1, movie2, ratings_matrix=None):
    """Calculate similarity between two movies based on features, genres, cast, and ratings."""
    # Calculate numerical feature similarity using cosine similarity
    numerical_features_1 = movie1.to_feature_vector()
    numerical_features_2 = movie2.to_feature_vector()
    numerical_similarity = cosine_similarity(numerical_features_1, numerical_features_2)

    # Calculate Jaccard similarity for genres
    genres_similarity = jaccard_similarity(set(movie1.genres), set(movie2.genres))

    # Calculate Jaccard similarity for cast
    cast_similarity = jaccard_similarity(set(movie1.cast), set(movie2.cast))

    # Calculate ratings similarity (if ratings_matrix is provided)
    if ratings_matrix is not None:
        # Get the rating vectors for both movies - handle missing ratings with 0s
        movie_1_ratings = ratings_matrix[movie1.title].values if movie1.title in ratings_matrix.columns else np.zeros(
            len(ratings_matrix))
        movie_2_ratings = ratings_matrix[movie2.title].values if movie2.title in ratings_matrix.columns else np.zeros(
            len(ratings_matrix))

        # Calculate the cosine similarity for ratings
        ratings_similarity = cosine_similarity(movie_1_ratings, movie_2_ratings)
    else:
        # If ratings_matrix is not provided, set ratings similarity to 0
        ratings_similarity = 0

    # Combine all similarities - Adjusted the weight (you can adjust the weights of each similarity if needed)
    total_similarity = (6 / 9) * numerical_similarity + (1 / 9) * genres_similarity + (1 / 9) * cast_similarity + (
                1 / 9) * ratings_similarity
    return total_similarity

def main():
    # Load and prepare the ratings data
    train_data = load_data('../../data/movie/raw/train.csv')
    ratings_matrix = create_ratings_matrix(train_data)

    # Initialize the TMDB client
    client = Client()

    # Get two movie objects using their IDs (for example, 62 and 63)
    movie_62 = client.get_movie(62)
    movie_63 = client.get_movie(63)

    # Calculate the similarity between these two movies based on features, genres, cast, and ratings
    similarity = calculate_movie_similarity(movie_62, movie_63, ratings_matrix)

    # Print the similarity score between the two movies based on all criteria (features, genres, cast, ratings)
    print(f"Total similarity between Movie 62 and Movie 63 based on features, genres, cast, and ratings: {similarity}")

if __name__ == "__main__":
    main()


