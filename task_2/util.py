import logging
import pathlib
from abc import ABC, abstractmethod

import pandas as pd

from data.movie import train, task as task_data
from movie import Movie
from movie.tmdb.client import Client
from task_2.classification.decision_tree import DecisionTree
from task_2.classification.random_forest import RandomForestClassifier

_tmdb_client = Client()

submission_dir = pathlib.Path(__file__).parent / 'submission'


def get_training_data_for_user(user_id: int) -> (list[Movie], list[int]):
    """get list of watched movies and ratings for a given user"""
    watched_movies_ratings = train[train['UserID'] == user_id]
    labels = watched_movies_ratings['Rating'].values.tolist()
    movies = [_tmdb_client.get_movie(movie_id) for movie_id in watched_movies_ratings['MovieID'].unique()]
    return movies, labels


class SubmissionGenerator(ABC):

    def __init__(self, submission_file_name: str):
        assert submission_file_name.endswith('.csv'), "Submission file name must end with .csv"
        self.submission_path = submission_dir / submission_file_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

    @abstractmethod
    def create_fitted_classifier(self, movies: list[Movie], labels: list[int]) -> DecisionTree | RandomForestClassifier:
        pass

    @abstractmethod
    def predict(self, classifier: DecisionTree | RandomForestClassifier, movies: list[Movie]) -> list[int]:
        pass

    def run(self):
        task: pd.DataFrame = task_data.copy()
        for user_id in task['UserID'].unique():
            movies, labels = get_training_data_for_user(user_id)
            self.logger.debug(f"Generating tree for user {user_id} from {len(movies)} movies")
            classifier = self.create_fitted_classifier(movies, labels)

            movie_ids_to_predict = task[task['UserID'] == user_id]['MovieID'].values.tolist()
            movies_to_predict = [_tmdb_client.get_movie(movie_id) for movie_id in movie_ids_to_predict]
            self.logger.debug(f"Generating prediction for user {user_id}, for {len(movies_to_predict)} movies")
            predictions = self.predict(classifier, movies_to_predict)

            task.loc[task['UserID'] == user_id, 'Rating'] = predictions
            assert task[task['UserID'] == user_id][
                       'Rating'].isna().sum() == 0, "There are still unpredicted movies in task"
            self.logger.info(f"Generated {len(predictions)} predictions for user {user_id}")

        assert task['Rating'].isna().sum() == 0, "There are still unpredicted movies in task"

        # cast the ratings to integer
        task['Rating'] = task['Rating'].astype(int)

        if self.submission_path.exists():
            self.logger.warning(f"Submission file already exists at {self.submission_path}. It will be overwritten.")
        if not submission_dir.exists():
            submission_dir.mkdir()
        task.to_csv(self.submission_path.absolute(), index=False, header=False, sep=';')
        self.logger.info(f"Successfully generated submission file at {self.submission_path}.")
