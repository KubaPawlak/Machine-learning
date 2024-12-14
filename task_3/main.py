import logging
from pathlib import Path
from typing import NewType, Callable

import pandas as pd

from data.movie import train, task
from task_3.prediction import predict_rating
from task_3.similarity import similarity_function
from util import submission
from util.validation import Validator

logger = logging.getLogger(__name__)

SUBMISSION_PATH = Path(__file__).parent / 'submission/submission.csv'

_RatingMatrix = NewType("RatingMatrix", pd.DataFrame)


class UserSimilarityModel(submission.Model[_RatingMatrix]):
    def __init__(self, training_data: pd.DataFrame):
        super().__init__(training_data, per_user=False)
        self.similarity_fn: Callable[[pd.Series, pd.Series], float] = similarity_function

    def create_model(self, training_data: pd.DataFrame) -> _RatingMatrix:
        pivot = training_data.pivot(index='UserID', columns='MovieID', values='Rating')
        return _RatingMatrix(pivot)

    def predict(self, model: _RatingMatrix, user_ids: list[int], movie_ids: list[int]) -> list[int]:
        predicted_ratings = []
        rating_matrix = model

        for user, movie in zip(user_ids, movie_ids):
            predicted_rating = predict_rating(user, movie, self.similarity_fn,
                                              rating_matrix)
            predicted_rating = int(round(predicted_rating))
            assert 0 <= predicted_rating <= 5
            predicted_ratings.append(predicted_rating)

        return predicted_ratings


def main():
    logging.basicConfig(level=logging.INFO)

    model = UserSimilarityModel(train)
    validator = Validator(model)
    validator.validation_set_accuracy()

    # generate submission
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.generate_submission(task).to_csv(SUBMISSION_PATH, index=False, sep=';')

    logger.info(f"Predictions saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
