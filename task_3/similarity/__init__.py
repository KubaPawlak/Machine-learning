import logging

import numpy as np
import pandas as pd

_similarity_cache: dict[(int, int), int] = dict()

logger = logging.getLogger(__name__)


def clear_cache():
    global _similarity_cache
    _similarity_cache = dict()
    logger.debug("Similarity cache cleared")


def similarity_function(user_1: pd.Series, user_2: pd.Series, min_common=3, pearson_weight=0.7):
    """
     Computes a similarity score between two users based on their movie ratings.

     This function uses a combination of Pearson correlation and Cosine similarity to assess how similar
     two users are in terms of their movie preferences. The similarity is calculated only for movies that
     both users have rated. If the number of common ratings is below a specified (min_common),
     the function returns a similarity score of 0 to ensure meaningful comparisons.

     The final similarity score is a weighted average of the two similarity metrics, where the weight
     for Pearson correlation can be adjusted using the pearson_weight parameter. A damping factor is
     applied based on the number of common ratings to reduce the influence of users with fewer shared ratings.

     Parameters:
         user_1: Ratings given by the first user.
         user_2: Ratings given by the second user.
         min_common: Minimum number of common ratings required to compute similarity.
         pearson_weight: Weighting factor for the Pearson similarity in the final score.
     """
    if (cached_value := _similarity_cache.get((user_1.name, user_2.name)) or
                        _similarity_cache.get((user_2.name, user_1.name))) is not None:
        logger.debug(f"Similarity for users {user_1.name}, {user_2.name} retrieved from cache: {cached_value}")
        return cached_value

    common_movies = ~user_1.isna() & ~user_2.isna()
    common_count = common_movies.sum()

    if common_count < min_common:
        _similarity_cache[(user_1.name, user_2.name)] = 0
        return 0  # Insufficient overlap

    user_1_common = user_1[common_movies]
    user_2_common = user_2[common_movies]

    # Mean-center the ratings
    user_1_centered = user_1_common - user_1_common.mean()
    user_2_centered = user_2_common - user_2_common.mean()

    # Pearson Similarity
    numerator = (user_1_centered * user_2_centered).sum()
    denominator = np.sqrt((user_1_centered ** 2).sum()) * np.sqrt((user_2_centered ** 2).sum())

    pearson_similarity = numerator / denominator if denominator != 0 else 0

    # Cosine Similarity
    cosine_numerator = np.dot(user_1_centered, user_2_centered)
    cosine_denominator = np.linalg.norm(user_1_centered) * np.linalg.norm(user_2_centered)

    cosine_similarity = cosine_numerator / cosine_denominator if cosine_denominator != 0 else 0

    combined_similarity_score = (pearson_weight * pearson_similarity + (1 - pearson_weight) * cosine_similarity)

    # Apply damping factor
    combined_similarity_score *= common_count / (common_count + 1)

    _similarity_cache[(user_1.name, user_2.name)] = combined_similarity_score
    logger.debug(f"Similarity for {user_1.name} and {user_2.name} calculated: {combined_similarity_score:.4f}")
    return combined_similarity_score
