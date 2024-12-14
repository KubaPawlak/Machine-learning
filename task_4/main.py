import logging
import time
from typing import Tuple

import pandas as pd

import data.movie
from data.movie import train
from task_4.model import Model, UserId, MovieId
from util import submission, validation


class CollaborativeFilteringModel(submission.Model[Model]):

    def create_model(self, training_data: pd.DataFrame) -> Model:
        model = Model(training_data, n_features=15)
        model.train(learning_rate=1e-3, regularization_parameter=1e-2, epochs=5_000)
        return model

    def predict(self, model: Model, user_ids: list[int], movie_ids: list[int]) -> list[int]:
        results = []
        for user, movie in zip(user_ids, movie_ids):
            user_id = UserId(user)
            movie_id = MovieId(movie)
            results.append(model(user_id, movie_id))
        return results


def report_scores(train_scores: Tuple[float, float] | None = None,
                  val_scores: Tuple[float, float] | None = None,
                  cross_scores: Tuple[float, float] | None = None) -> None:
    # flush all pending logs
    for h in logging.getLogger().handlers:
        h.flush()
    if train_scores is not None:
        print("===   Training set accuracies   ===")
        print(f"Correct: {train_scores[0]:.2f}")
        print(f"One off: {train_scores[1]:.2f}")
    if val_scores is not None:
        print("===  Validation set accuracies  ===")
        print(f"Correct: {val_scores[0]:.2f}")
        print(f"One off: {val_scores[1]:.2f}")
    if cross_scores is not None:
        print("=== Cross-validation accuracies ===")
        print(f"Correct: {cross_scores[0]:.2f}")
        print(f"One off: {cross_scores[1]:.2f}")


def _main():
    logging.basicConfig(level=logging.INFO)
    model = CollaborativeFilteringModel(train)
    validator = validation.Validator(model)
    report_scores(
        train_scores=validator.train_set_accuracy(),
        val_scores=validator.validation_set_accuracy(),
        cross_scores=validator.k_fold_cross_validation(k=5),
    )
    time.sleep(0.5)
    s = model.generate_submission(data.movie.task)
    s.to_csv("submission.csv", index=False, header=False, sep=';')


if __name__ == '__main__':
    _main()
