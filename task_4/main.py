import logging

import pandas as pd

from data.movie import train
from task_4.model import Model, UserId, MovieId
from util import submission, validation

class CollaborativeFilteringModel(submission.Model[Model]):

    def create_model(self, training_data: pd.DataFrame) -> Model:
        model = Model(training_data, n_features=20)
        model.train(learning_rate=0.001, regularization_parameter=0.1, epochs=100)
        return model

    def predict(self, model: Model, user_ids: list[int], movie_ids: list[int]) -> list[int]:
        results = []
        for user, movie in zip(user_ids, movie_ids):
            user_id = UserId(user)
            movie_id = MovieId(movie)
            results.append(model(user_id, movie_id))
        return results


def _main():
    logging.basicConfig(level=logging.DEBUG)
    model = CollaborativeFilteringModel(train)
    validator = validation.Validator(model)
    validator.k_fold_cross_validation()

if __name__ == '__main__':
    _main()