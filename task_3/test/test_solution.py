import logging
from typing import Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

from data.movie import train as whole_train
from task_3.main import predict_ratings
from task_3.similarity import similarity_function

logger = logging.getLogger(__name__)


def split_data(train_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_test, test_test = train_test_split(train_data, test_size=0.2, random_state=42)
    return train_test, test_test


def clear_ratings(data: pd.DataFrame, in_place: bool = False) -> pd.DataFrame:
    if not in_place:
        data = data.copy()
    data['Rating'] = np.nan
    return data


def calculate_metrics(y_true: list[float], y_pred: list[float]) -> float:
    assert len(y_true) == len(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    accuracy = accuracy_score(y_true, y_pred)

    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    logger.info(f"Accuracy: {round(accuracy * 100)}%")
    return accuracy


def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting the process...")

    train, test = split_data(whole_train)

    task = clear_ratings(test, in_place=False)

    rating_matrix = train.pivot(index='UserID', columns='MovieID', values='Rating')

    logger.info("Predicting ratings...")
    predicted_ratings = predict_ratings(task, similarity_function, rating_matrix)

    logger.info("Calculating accuracy...")
    calculate_metrics(test['Rating'], predicted_ratings)


if __name__ == "__main__":
    main()
