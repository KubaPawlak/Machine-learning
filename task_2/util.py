import logging
import multiprocessing as mp
import pathlib
from abc import ABC, abstractmethod
from itertools import repeat

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

    def __init__(self, submission_file_name: str | None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        if submission_file_name is None:
            self.logger.warning("Submission file not set. The results will not be saved!")
            self.submission_path = None
        else:
            assert submission_file_name.endswith('.csv'), "Submission file name must end with .csv"
            self.submission_path: pathlib.Path | None = submission_dir / submission_file_name

    @abstractmethod
    def create_fitted_classifier(self, movies: list[Movie], labels: list[int]) -> DecisionTree | RandomForestClassifier:
        pass

    @abstractmethod
    def predict(self, classifier: DecisionTree | RandomForestClassifier, movies: list[Movie]) -> list[int]:
        pass

    def run(self):
        task: pd.DataFrame = task_data.copy()
        num_users = train['UserID'].nunique()
        for j, user_id in enumerate(train['UserID'].unique()):
            self.logger.info(f"Running user {user_id:<4} ({j + 1}/{num_users})")
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

        if self.submission_path is not None:
            if self.submission_path.exists():
                self.logger.warning(
                    f"Submission file already exists at {self.submission_path}. It will be overwritten.")
            if not submission_dir.exists():
                submission_dir.mkdir()
            task.to_csv(self.submission_path.absolute(), index=False, header=False, sep=';')
            self.logger.info(f"Successfully generated submission file at {self.submission_path}.")


class Validator:
    def __init__(self, submission_generator: SubmissionGenerator, num_runs: int = 3):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.submission_generator = submission_generator
        self.num_runs = num_runs

    @staticmethod
    def calculate_metrics(y_true, y_pred) -> (float, float):
        """Calculate the fraction of correct predictions, and one-off predictions"""
        assert len(y_true) == len(y_pred), "y_true and y_pred must have same length"
        correct_count = 0
        one_off_count = 0
        total_count = len(y_true)
        for (true, pred) in zip(y_true, y_pred):
            if true == pred:
                correct_count += 1
            elif abs(true - pred) == 1:
                one_off_count += 1

        return correct_count / total_count, one_off_count / total_count

    def run_user(self, user_id):
        movies, labels = get_training_data_for_user(user_id)
        movies_train, movies_test, labels_train, labels_test = train_test_split(movies, labels)
        classifier = self.submission_generator.create_fitted_classifier(movies_train, labels_train)
        predictions = self.submission_generator.predict(classifier, movies_test)
        accuracy, one_off_accuracy = Validator.calculate_metrics(labels_test, predictions)
        return accuracy, one_off_accuracy

    @staticmethod
    def _user_func(validator, user_id: int):
        validator.logger.debug(f"Running tests for user {user_id}")
        result = validator.run_user(user_id)
        validator.logger.debug(f"User {user_id} done")
        return result

    def run_parallel(self):
        accuracies = []
        one_off_accuracies = []

        for i in range(self.num_runs):
            self.logger.info(f"Running iteration {i + 1}/{self.num_runs}")
            user_ids = list(train['UserID'].unique())
            with mp.Pool(mp.cpu_count()) as pool:
                results: list[(float, float)] = pool.starmap(Validator._user_func, zip(repeat(self), user_ids))
            run_accuracies, run_one_off_accuracies = tuple(map(lambda x: list(x), zip(*results)))

            accuracies.append(np.mean(run_accuracies))
            one_off_accuracies.append(np.mean(run_one_off_accuracies))

        print(f"Average accuracy: {np.mean(accuracies).round(2) * 100}%")
        print(f"Average one-off accuracy: {np.mean(one_off_accuracies).round(2) * 100}%")

    def run_sequential(self):
        accuracies = []
        one_off_accuracies = []
        num_users = train['UserID'].nunique()
        for i in range(self.num_runs):

            self.logger.info(f"Running iteration {i + 1}/{self.num_runs}")
            run_accuracies = []
            run_one_off_accuracies = []
            for j, user_id in enumerate(train['UserID'].unique()):
                self.logger.debug(f"Running tests for user {user_id:<4} ({j + 1}/{num_users})")
                accuracy, one_off_accuracy = self.run_user(user_id)
                run_accuracies.append(accuracy)
                run_one_off_accuracies.append(one_off_accuracy)

            accuracies.append(np.mean(run_accuracies))
            one_off_accuracies.append(np.mean(run_one_off_accuracies))

        print(f"Average accuracy: {np.mean(accuracies).round(2) * 100}%")
        print(f"Average one-off accuracy: {np.mean(one_off_accuracies).round(2) * 100}%")
