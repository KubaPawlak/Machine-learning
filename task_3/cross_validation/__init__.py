import numpy as np
from sklearn.model_selection import KFold

from task_3.prediction import predict_rating


def kfold_cross_validation(data, similarity_func, similarity_cache, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []

    for train_indices, val_indices in kf.split(data):
        train_data = data.iloc[train_indices]
        val_data = data.iloc[val_indices]

        train_matrix = train_data.pivot(index='UserID', columns='MovieID', values='Rating')

        predictions = []
        true_ratings = []
        for _, row in val_data.iterrows():
            target_user = row['UserID']
            target_movie = row['MovieID']
            true_rating = row['Rating']

            predicted_rating = predict_rating(
                target_user, target_movie, similarity_func, similarity_cache, train_matrix
            )
            predictions.append(predicted_rating)
            true_ratings.append(true_rating)

        mse = np.mean((np.array(predictions) - np.array(true_ratings)) ** 2)
        mse_scores.append(mse)

    return np.mean(mse_scores), np.std(mse_scores)