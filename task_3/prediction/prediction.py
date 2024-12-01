def predict_rating(target_user, target_movie, similarity_func, similarity_cache, rating_matrix):
    movie_ratings = rating_matrix[target_movie]
    rated_by_others = movie_ratings.dropna()

    similarities = []
    ratings = []

    for other_user in rated_by_others.index:
        similarity = similarity_func(rating_matrix.loc[target_user], rating_matrix.loc[other_user], similarity_cache)
        if similarity > 0:  # Consider only positive similarities
            similarities.append(similarity)
            ratings.append(movie_ratings[other_user])

    if not similarities:
        print(f"No positive similarities found for user {target_user} and movie {target_movie}. Using average rating.")
        return movie_ratings.mean()  # Fallback: average rating for the movie

    weighted_sum = sum(sim * rat for sim, rat in zip(similarities, ratings))
    normalization_factor = sum(abs(sim) for sim in similarities)

    predicted_rating = weighted_sum / normalization_factor
    print(f"Predicted rating for user {target_user} and movie {target_movie}: {predicted_rating:.4f}")
    return predicted_rating
