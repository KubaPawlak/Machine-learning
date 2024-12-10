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


def report_scores(train_scores, val_scores, cross_scores):
    print("===   Training set accuracies   ===")
    print(f"Correct: {train_scores[0]:.2f}")
    print(f"One off: {train_scores[1]:.2f}")
    print("===  Validaiton set accuracies  ===")
    print(f"Correct: {val_scores[0]:.2f}")
    print(f"One off: {val_scores[1]:.2f}")
    print("=== Cross-validation accuracies ===")
    print(f"Correct: {cross_scores[0]:.2f}")
    print(f"One off: {cross_scores[1]:.2f}")

def _main():
    logging.basicConfig(level=logging.INFO)
    model = CollaborativeFilteringModel(train)
    validator = validation.Validator(model)
    train_set_scores = validator.train_set_accuracy()
    validation_scores = validator.validation_set_accuracy()
    cross_validation_scores = validator.k_fold_cross_validation()
    report_scores(train_set_scores, validation_scores, cross_validation_scores)

if __name__ == '__main__':
    _main()