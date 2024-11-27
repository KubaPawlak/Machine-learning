import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from data.movie import train, task
from movie import Movie
from movie.tmdb.client import Client
from task_2.classification.decision_tree import DecisionTree

logging.basicConfig(level=logging.INFO)
tmdb_client = Client()
_RESULT_FILE = Path('submission_tree.csv').absolute()


def get_movie_features(unique_movie_ids: Iterable[int]) -> dict[int, Movie]:
    movie_features = {movie_id: tmdb_client.get_movie(movie_id) for movie_id in unique_movie_ids}
    return movie_features


def train_decision_tree(train_data: pd.DataFrame, movie_features: dict[int, Movie]) -> DecisionTree:
    train_movies: list[Movie] = [movie_features[movie_id] for movie_id in train_data["MovieID"]]
    train_ratings: list[int] = train_data["Rating"].tolist()

    decision_tree = DecisionTree(max_depth=10)
    decision_tree.fit(train_movies, train_ratings)
    logging.info(f"Trained decision tree for {len(train_data)} data points.")
    return decision_tree


def predict_ratings(task_data: pd.DataFrame, movie_features: dict[int, Movie], classifier: DecisionTree):
    task_movies = [movie_features[movie_id] for movie_id in task_data["MovieID"]]
    predicted_ratings = classifier.predict(task_movies)
    logging.info(f"Predicted ratings for {len(task_data)} task entries.")
    return predicted_ratings


def prepare_submission_rows(task_data: pd.DataFrame, predicted_ratings: list[int]):
    submission_rows = []
    for row, predicted_rating in zip(task_data.itertuples(), predicted_ratings):
        submission_rows.append(f"{int(row.ID)};{int(row.UserID)};{int(row.MovieID)};{int(predicted_rating)}")
    logging.info(f"Prepared {len(submission_rows)} rows.")
    return submission_rows


def save_submission(submission_rows):
    with open(_RESULT_FILE, "w") as submission_file:
        submission_file.write("\n".join(submission_rows))
    logging.info(f"Saved to {_RESULT_FILE}.")


def main():
    unique_movie_ids = train["MovieID"].unique()
    movie_features = get_movie_features(unique_movie_ids)
    submission_rows = []

    for user_id in task["UserID"].unique():
        user_train_data = train[train["UserID"] == user_id]
        user_task_data = task[task["UserID"] == user_id]
        decision_tree = train_decision_tree(user_train_data, movie_features)
        predicted_ratings = predict_ratings(user_task_data, movie_features, decision_tree)
        user_submission_rows = prepare_submission_rows(user_task_data, predicted_ratings)
        submission_rows.extend(user_submission_rows)

    save_submission(submission_rows)


if __name__ == "__main__":
    main()
