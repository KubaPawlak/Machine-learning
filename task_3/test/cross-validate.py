import logging
from typing import Callable, Tuple, Any

import numpy as np
import pandas as pd
from numpy import floating
from sklearn.model_selection import KFold

from data.movie import train
from task_3.prediction import predict_rating
from task_3.similarity import similarity_function

def k_fold_cross_validation(data: pd.DataFrame,
                            similarity_func: Callable,
                            k: int = 5) -> tuple[floating[Any], floating[Any]]:

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

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

        # Calculate accuracy: predictions are correct if they are within 0.5 of the true rating
        correct_predictions = sum(
            1 for pred, true in zip(predictions, true_ratings) if abs(pred - true) <= 0.5
        )
        accuracy = (correct_predictions / len(true_ratings)) * 100  # Convert to percentage
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)


def main():
    print("Starting k-fold cross-validation...")

    # Perform k-fold cross-validation
    mean_accuracy, std_accuracy = k_fold_cross_validation(train, similarity_function, k=5)
    print(f"Mean Accuracy from k-fold: {mean_accuracy:.2f}%, Standard Deviation: {std_accuracy:.2f}%")

if __name__ == "__main__":
    main()