import logging
from pathlib import Path

import pandas as pd

from data.movie import train, task
from task_3.prediction import predict_rating
from task_3.similarity import similarity_function

logger = logging.getLogger(__name__)

SUBMISSION_PATH = Path(__file__).parent / 'submission/submission.csv'

def predict_ratings(users_and_movies: pd.DataFrame, similarity_func, rating_matrix: pd.DataFrame) -> list[int]:
    predicted_ratings = []

    for _, row in users_and_movies.iterrows():
        target_user = row['UserID']
        target_movie = row['MovieID']
        predicted_rating = predict_rating(target_user, target_movie, similarity_func,
                                          rating_matrix)
        predicted_rating = int(round(predicted_rating))
        assert 0 <= predicted_rating <= 5
        predicted_ratings.append(predicted_rating)

    return predicted_ratings

def main():
    logging.basicConfig(level=logging.INFO)

    rating_matrix = train.pivot(index='UserID', columns='MovieID', values='Rating')

    predicted_ratings = predict_ratings(task, similarity_function, rating_matrix)

    # Replace null values in task.csv with predicted ratings
    task['Rating'] = predicted_ratings
    assert task['Rating'].isna().sum() == 0

    SUBMISSION_PATH.parent.mkdir(parents=True,exist_ok=True)
    task.to_csv(SUBMISSION_PATH, index=False, sep=';')

    logger.info(f"Predictions saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()