import logging

import pandas as pd

import data.movie as data
import movie.tmdb.client
from task_2.classification.decision_tree import DecisionTree
from task_2.classification.random_forest import RandomForestClassifier
from util import submission, validation


class DecisionTreeModel(submission.Model[DecisionTree]):
    def __init__(self, training_data: pd.DataFrame):
        super().__init__(training_data, per_user=True)
        tmdb_client = movie.tmdb.client.Client()
        self.all_movies = {id_: tmdb_client.get_movie(id_) for id_ in training_data['MovieID'].unique()}

    def create_model(self, training_data: pd.DataFrame) -> DecisionTree:
        movies = [self.all_movies[id_] for id_ in training_data['MovieID'].values]
        labels = training_data['Rating'].values.tolist()

        tree = DecisionTree(max_depth=5)
        tree.fit(movies, labels)
        return tree

    def predict(self, model: DecisionTree, user_ids: list[int], movie_ids: list[int]) -> list[int]:
        assert pd.Series(user_ids).nunique() == 1

        movies_to_predict = [self.all_movies[id_] for id_ in movie_ids]
        return model.predict(movies_to_predict)


class RandomForestModel(submission.Model[RandomForestClassifier]):
    def __init__(self, training_data: pd.DataFrame):
        super().__init__(training_data, per_user=True)
        tmdb_client = movie.tmdb.client.Client()
        self.all_movies = {id_: tmdb_client.get_movie(id_) for id_ in training_data['MovieID'].unique()}

    def create_model(self, training_data: pd.DataFrame) -> RandomForestClassifier:
        movies = [self.all_movies[id_] for id_ in training_data['MovieID'].values]
        labels = training_data['Rating'].values.tolist()

        forest = RandomForestClassifier(num_trees=10, num_features=3, max_depth=5)
        forest.fit(movies, labels)
        return forest

    def predict(self, model: RandomForestClassifier, user_ids: list[int], movie_ids: list[int]) -> list[int]:
        assert pd.Series(user_ids).nunique() == 1

        movies_to_predict = [self.all_movies[id_] for id_ in movie_ids]
        return model.predict(movies_to_predict)


def _main():
    logging.basicConfig(level=logging.INFO)

    model = DecisionTreeModel(data.train)
    validator = validation.Validator(model)
    validator.validation_set_accuracy()


if __name__ == '__main__':
    _main()
