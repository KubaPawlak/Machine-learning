import logging
from abc import ABC, abstractmethod

import pandas as pd


class Model[TModel](ABC):
    def __init__(self, training_data: pd.DataFrame, per_user: bool = False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.train_set = training_data
        self.per_user = per_user

    def select_training_data(self, user_id: int | None = None) -> pd.DataFrame:
        if user_id is not None:
            return self.train_set[self.train_set['UserID'] == user_id]
        return self.train_set

    @abstractmethod
    def train(self, training_data: pd.DataFrame) -> TModel:
        pass

    @abstractmethod
    def predict(self,
                model: TModel,
                user_ids: list[int],
                movie_ids: list[int]
                ) -> list[int]:
        pass

    def generate_submission(self, task: pd.DataFrame) -> pd.DataFrame:
        task: pd.DataFrame = task.copy()
        if self.per_user:
            result = self._generate_submission_per_user(task)
        else:
            result = self._generate_submission_for_all(task)
        assert result['Rating'].isna().sum() == 0, "There are still unpredicted movies in task"
        return result

    def _generate_submission_per_user(self, task: pd.DataFrame) -> pd.DataFrame:
        num_users = self.train_set['UserID'].nunique()
        for j, user_id in enumerate(self.train_set['UserID'].unique()):
            self.logger.info(f"Running user {user_id:<4} ({j + 1}/{num_users})")
            user_train = self.select_training_data(user_id)
            self.logger.debug(f"Training model for user {user_id} from {len(user_train)} movies")
            model = self.train(user_train)

            movies_to_predict = task[task['UserID'] == user_id]['MovieID'].values.tolist()
            self.logger.debug(f"Generating prediction for user {user_id}, for {len(movies_to_predict)} movies")
            predictions = self.predict(model, [user_id for _ in movies_to_predict], movies_to_predict)

            task.loc[task['UserID'] == user_id, 'Rating'] = predictions
            assert task[task['UserID'] == user_id][
                       'Rating'].isna().sum() == 0, "There are still unpredicted movies in task"
            self.logger.info(f"Generated {len(predictions)} predictions for user {user_id}")

        return task

    def _generate_submission_for_all(self, task: pd.DataFrame) -> pd.DataFrame:
        user_ids: list[int] = task['UserID'].tolist()
        movie_ids: list[int] = task['MovieID'].tolist()

        train = self.select_training_data()
        self.logger.debug(f"Training model from {len(train)} data points")
        model = self.train(train)
        self.logger.debug(f"Generating predictions")
        predictions = self.predict(model, user_ids, movie_ids)
        task['Rating'] = predictions
        self.logger.info(f"Generated {len(predictions)} predictions")
        return task
