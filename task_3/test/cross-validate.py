import logging

import numpy as np
from sklearn.model_selection import KFold

from data.movie import train
from task_3.prediction import predict_rating
from task_3.similarity import similarity_function

def k_fold_cross_validation(data, similarity_func, k=5):
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
                target_user, target_movie, similarity_func, train_matrix
            )
            predictions.append(predicted_rating)
            true_ratings.append(true_rating)

        mse = np.mean((np.array(predictions) - np.array(true_ratings)) ** 2)
        mse_scores.append(mse)

    return np.mean(mse_scores), np.std(mse_scores)

def _main():
    logging.basicConfig(level=logging.INFO)
    mse_mean, mse_stddev = k_fold_cross_validation(train, similarity_function, k=5)
    print(f"MSE = {mse_mean:.2f} (\u03c3={mse_stddev:.2f})")

if __name__ == '__main__':
    _main()