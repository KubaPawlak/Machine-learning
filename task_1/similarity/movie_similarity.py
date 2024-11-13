import pandas as pd

from data.movie import train
from task_1.movie import Movie
from .cosine_similarity import cosine_similarity
from .jaccard_similarity import jaccard_similarity

def create_rating_matrix(train_data: pd.DataFrame) -> pd.DataFrame:
    return train_data.pivot_table(index='UserID', columns='MovieID', values='Rating', fill_value=0)

_global_rating_matrix = create_rating_matrix(train)

def _calculate_rating_similarity(movie_1: Movie, movie_2: Movie, rating_matrix) -> float:
    assert rating_matrix is not None
    # Get the rating vectors for both movies
    movie_1_ratings = rating_matrix[movie_1.movie_id].values
    movie_2_ratings = rating_matrix[movie_2.movie_id].values
    # Calculate cosine similarity for ratings
    return cosine_similarity(movie_1_ratings, movie_2_ratings)


def calculate_movie_similarity(movie1: Movie, movie2: Movie,
                               rating_matrix: pd.DataFrame | None = _global_rating_matrix) -> float:
    """Calculate similarity between two movies based on similarity, genres, cast, and ratings."""

    # Calculate numerical feature similarity using cosine similarity
    numerical_features_1 = movie1.to_feature_vector()
    numerical_features_2 = movie2.to_feature_vector()
    numerical_similarity = cosine_similarity(numerical_features_1, numerical_features_2)

    # Calculate Jaccard similarity for genres
    genres_similarity = jaccard_similarity(set(movie1.genres), set(movie2.genres))

    # Calculate Jaccard similarity for cast
    cast_similarity = jaccard_similarity(set(movie1.cast), set(movie2.cast))

    # Calculate rating similarity if ratings_matrix is provided
    ratings_similarity: float | None = None
    if rating_matrix is not None:
        ratings_similarity = _calculate_rating_similarity(movie1, movie2, rating_matrix)

    # Combine all similarities with adjusted weights
    total_similarity = (
            (6 / 9) * numerical_similarity
            + (1 / 9) * genres_similarity
            + (1 / 9) * cast_similarity
            + (1 / 9) * ratings_similarity
    )
    return total_similarity
