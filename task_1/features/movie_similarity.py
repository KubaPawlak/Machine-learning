from data.movie import train
from task_1.features import cosine_similarity, jaccard_similarity
from task_1.movie import Movie

ratings_matrix = train.pivot_table(index='UserID', columns='MovieID', values='Rating', fill_value=0)


def calculate_movie_similarity(movie1: Movie, movie2: Movie) -> float:
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
    if ratings_matrix is not None:

        # Get the rating vectors for both movies based on MovieID (TMDBID)
        movie_1_ratings = ratings_matrix[movie1.movie_id].values
        movie_2_ratings = ratings_matrix[movie2.movie_id].values
        # Calculate cosine similarity for ratings
        ratings_similarity = cosine_similarity(movie_1_ratings, movie_2_ratings)

    else:
        # If no ratings matrix or mapping is provided, default to 0 for rating similarity
        ratings_similarity = 0

        # Combine all similarities with adjusted weights
    total_similarity = (6 / 9) * numerical_similarity + (1 / 9) * genres_similarity + (1 / 9) * cast_similarity + (
            1 / 9) * ratings_similarity
    return total_similarity
