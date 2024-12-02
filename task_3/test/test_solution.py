import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from data.movie import train
from task_3.prediction import predict_rating
from task_3.similarity import similarity_function

logger = logging.getLogger(__name__)

def split_data(train_data):
    train_test, test_test = train_test_split(train_data, test_size=0.2, random_state=42)
    return train_test, test_test


def replace_ratings_with_nan(test_test):
    test_test_with_nulls = test_test.copy()
    test_test_with_nulls['Rating'] = np.nan
    return test_test_with_nulls


def predict_for_test_set(test_set, similarity_func, rating_matrix_train):
    predicted_ratings = []

    for _, row in test_set.iterrows():
        target_user = row['UserID']
        target_movie = row['MovieID']
        predicted_rating = predict_rating(target_user, target_movie, similarity_func,
                                          rating_matrix_train)
        predicted_ratings.append(predicted_rating)

    return predicted_ratings

def calculate_accuracy(real_ratings, test_test_with_nulls, predicted_ratings):
    test_test_with_nulls['Rating'] = predicted_ratings

    comparison_df = pd.merge(real_ratings, test_test_with_nulls[['UserID', 'MovieID', 'Rating']], on=['UserID', 'MovieID'], how='left')

    # Calculate MAE and RMSE
    mae = mean_absolute_error(comparison_df['Rating_x'], comparison_df['Rating_y'])
    rmse = np.sqrt(mean_squared_error(comparison_df['Rating_x'], comparison_df['Rating_y']))

    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Calculate accuracy as percentage of correct predictions
    correct_predictions = comparison_df[comparison_df['Rating_x'] == comparison_df['Rating_y']].shape[0]
    total_predictions = comparison_df.shape[0]
    accuracy_percentage = (correct_predictions / total_predictions) * 100

    logger.info(f"Accuracy: {accuracy_percentage:.2f}%")


def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting the process...")

    train_test, test_test = split_data(train)

    real_ratings = test_test[['UserID', 'MovieID', 'Rating']].copy()

    test_test_with_nulls = replace_ratings_with_nan(test_test)

    rating_matrix_train = train_test.pivot(index='UserID', columns='MovieID', values='Rating')

    logger.info("Predicting ratings for test_test...")
    predicted_ratings = predict_for_test_set(test_test_with_nulls, similarity_function, rating_matrix_train)

    logger.info("Calculating accuracy...")
    calculate_accuracy(real_ratings, test_test_with_nulls, predicted_ratings)


if __name__ == "__main__":
    main()