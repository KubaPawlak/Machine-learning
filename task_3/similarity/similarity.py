from typing import Dict

import numpy as np
import pandas as pd


def similarity_function(user1: pd.Series,
                        user2: pd.Series,
                        similarity_cache: Dict[tuple, float],
                        min_common: int = 3,
                        pearson_weight: float = 0.7) -> float:
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
         user1: Ratings given by the first user.
         user2: Ratings given by the second user.
         similarity_cache: A cache to store previously computed similarity scores for efficiency.
         min_common: Minimum number of common ratings required to compute similarity.
         pearson_weight: Weighting factor for the Pearson similarity in the final score.
     """
    if (user1.name, user2.name) in similarity_cache:
        return similarity_cache[(user1.name, user2.name)]

    common_movies = ~user1.isna() & ~user2.isna()
    common_count = common_movies.sum()

    if common_count < min_common:
        similarity_cache[(user1.name, user2.name)] = 0
        return 0  # Insufficient overlap

    user1_common = user1[common_movies]
    user2_common = user2[common_movies]

    # Mean-center the ratings
    user1_centered = user1_common - user1_common.mean()
    user2_centered = user2_common - user2_common.mean()

    # Pearson Similarity
    numerator = (user1_centered * user2_centered).sum()
    denominator = np.sqrt((user1_centered ** 2).sum()) * np.sqrt((user2_centered ** 2).sum())

    pearson_similarity = numerator / denominator if denominator != 0 else 0

    # Cosine Similarity
    cosine_numerator = np.dot(user1_centered, user2_centered)
    cosine_denominator = np.linalg.norm(user1_centered) * np.linalg.norm(user2_centered)

    cosine_similarity = cosine_numerator / cosine_denominator if cosine_denominator != 0 else 0

    combined_similarity_score = (pearson_weight * pearson_similarity + (1 - pearson_weight) * cosine_similarity)

    # Apply damping factor
    combined_similarity_score *= common_count / (common_count + 1)

    similarity_cache[(user1.name, user2.name)] = combined_similarity_score
    print(f"Similarity for {user1.name} and {user2.name} calculated: {combined_similarity_score:.4f}")
    return combined_similarity_score