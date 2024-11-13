from collections.abc import Iterable

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.movie import train
from task_1.movie import Movie
from .vector import cosine_similarity, euclidean_similarity, manhattan_similarity
from .set import jaccard_similarity

scaler = MinMaxScaler()


def fit_scaler(movies: Iterable[Movie]) -> None:
    movies_features = [m.to_feature_vector() for m in movies]
    scaler.fit(movies_features)


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
                               rating_matrix: pd.DataFrame | None = _global_rating_matrix,
                               scalar_similarity_part: float = (6.0 / 10),
                               genres_similarity_part: float = (1.0 / 10),
                               cast_similarity_part: float = (1.0 / 10),
                               directors_similarity_part: float = (1.0 / 10),
                               ratings_similarity_part: float = (1.0 / 10)) -> float:
    """Calculate similarity between two movies based on similarity, genres, cast, and ratings."""

    # Calculate numerical feature similarity using cosine similarity
    scalar_features_1 = movie_1.to_feature_vector()
    scalar_features_2 = movie_2.to_feature_vector()

    # Applying scaler for numerical features
    scalar_features_1 = scaler.transform([scalar_features_1])[0]
    scalar_features_2 = scaler.transform([scalar_features_2])[0]

    # Calculate Jaccard similarity for genres, cast and directors
    genres_similarity = jaccard_similarity(set(movie_1.genres),
                                           set(movie_2.genres)) if genres_similarity_part != 0 else 0
    cast_similarity = jaccard_similarity(set(movie_1.cast),
                                         set(movie_2.cast)) if cast_similarity_part != 0 else 0
    directors_similarity = jaccard_similarity(set(movie_1.director),
                                              set(movie_2.director)) if directors_similarity_part != 0 else 0

    if metric == 'cosine':
        scalar_features_similarity = cosine_similarity(scalar_features_1, scalar_features_2)
    elif metric == 'manhattan':
        scalar_features_similarity = manhattan_similarity(scalar_features_1, scalar_features_2)
    elif metric == 'euclidean':
        scalar_features_similarity = euclidean_similarity(scalar_features_1, scalar_features_2)
    else:
        scalar_features_similarity = cosine_similarity(scalar_features_1, scalar_features_2)
    assert scalar_features_similarity is not None

    # Calculate rating similarity if ratings_matrix is provided
    ratings_similarity: float = 0
    if rating_matrix is not None and ratings_similarity_part != 0:
        ratings_similarity = _calculate_rating_similarity(movie_1, movie_2, rating_matrix)

    # Combine all similarities with adjusted weights
    total_similarity = (
            scalar_similarity_part * scalar_features_similarity
            + genres_similarity_part * genres_similarity
            + cast_similarity_part * cast_similarity
            + directors_similarity_part * directors_similarity
            + ratings_similarity_part * ratings_similarity
    )
    return total_similarity
