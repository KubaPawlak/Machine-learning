import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.movie import train
from task_1.movie import Movie
from .cosine_similarity import cosine_similarity
from .jaccard_similarity import jaccard_similarity
from .manhattan_similarity import manhattan_distance


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


def calculate_movie_similarity(movie_1: Movie, movie_2: Movie, metric: str = 'cosine',
                               rating_matrix: pd.DataFrame | None = _global_rating_matrix) -> float:
    """Calculate similarity between two movies based on similarity, genres, cast, and ratings."""

    # Calculate numerical feature similarity using cosine similarity
    numerical_features_1 = movie_1.to_feature_vector()
    numerical_features_2 = movie_2.to_feature_vector()

    # Applying scaler for numerical features
    scaler = MinMaxScaler()
    scaled_features_1 = scaler.fit_transform([numerical_features_1])[0]
    scaled_features_2 = scaler.fit_transform([numerical_features_2])[0]

    # Calculate Jaccard similarity for genres, cast and directors
    genres_similarity = jaccard_similarity(set(movie_1.genres), set(movie_2.genres))
    cast_similarity = jaccard_similarity(set(movie_1.cast), set(movie_2.cast))
    directors_similarity = jaccard_similarity(set(movie_1.director), set(movie_2.director))

    if metric == 'cosine':
        numerical_similarity = cosine_similarity(scaled_features_1, scaled_features_2)
    elif metric == 'manhattan':
        numerical_similarity = manhattan_distance(scaled_features_1, scaled_features_2)
    else:
        numerical_similarity = cosine_similarity(scaled_features_1, scaled_features_2)

    # Calculate rating similarity if ratings_matrix is provided
    ratings_similarity: float | None = None
    if rating_matrix is not None:
        ratings_similarity = _calculate_rating_similarity(movie_1, movie_2, rating_matrix)

    # Combine all similarities with adjusted weights
    total_similarity = (
            (7 / 11) * numerical_similarity
            + (1 / 11) * genres_similarity
            + (1 / 11) * cast_similarity
            + (1 / 11) * directors_similarity
            + (1 / 11) * ratings_similarity
    )
    return total_similarity
