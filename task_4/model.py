from typing import NewType, Iterable

import numpy as np
import pandas as pd

UserId = NewType('UserId', int)
_UserIndex = NewType('UserIndex', int)
MovieId = NewType('MovieId', int)
_MovieIndex = NewType('MovieIndex', int)

_PGradients = NewType('PGradients', np.ndarray)
_XGradients = NewType('XGradients', np.ndarray)


class Model:

    def __init__(self, train: pd.DataFrame, n_features: int = 5):
        # Create the pivot table for ratings
        ratings = train.pivot(index='MovieID', columns='UserID', values='Rating')
        self.y = ratings.fillna(-1).to_numpy()

        # Create mappings directly from ratings.index and ratings.columns
        self.map_user = {UserId(user_id): _UserIndex(idx) for idx, user_id in enumerate(ratings.columns)}
        self.map_movie = {MovieId(movie_id): _MovieIndex(idx) for idx, movie_id in enumerate(ratings.index)}

        # Initialize parameters for collaborative filtering
        self.n_features = n_features
        self.x = np.random.rand(self.y.shape[0], n_features)  # Movie features
        self.p = np.random.rand(self.y.shape[1], n_features + 1)  # User parameters

    def _existing_ratings(self) -> Iterable[tuple[_MovieIndex, _UserIndex]]:
        return np.argwhere(self.y != -1)

    def _error(self, user: _UserIndex, movie: _MovieIndex) -> float:
        prediction = self._calculate_prediction(user, movie)
        actual = self._get_actual_rating(user, movie)
        assert actual is not None
        return prediction - actual

    def _loss(self) -> float:
        total_loss = 0.0
        for movie, user in self._existing_ratings():
            total_loss += 0.5 * self._error(user, movie) ** 2
        return total_loss

    def _compute_gradients(self, regularization_parameter:float=0.0) -> tuple[_PGradients, _XGradients]:
        dp = np.zeros_like(self.p)
        dx = np.zeros_like(self.x)

        for m, u in self._existing_ratings():
            error = self._error(u, m)
            dp[u, 0] += error  # bias term p_0
            dp[u, 1:] += error * self.x[m, :]  # user parameters p_i for i = 1,2,...N
            dx[m, :] += error * self.p[u, 1:]  # movie features x_i for i = 1,2,...N

        dp[:,1:] += regularization_parameter * self.p[:,1:]
        dx += regularization_parameter * self.x

        return _PGradients(dp), _XGradients(dx)

    def train(self, learning_rate=0.01, n_iterations=1000, **kwargs):
        for i in range(n_iterations):

            dp, dx = self._compute_gradients(**kwargs)
            self.p -= learning_rate * dp
            self.x -= learning_rate * dx

            if i % 10 == 0:
                print(f'Iteration: {i:4}, Loss: {self._loss():.2f}')

    def _get_actual_rating(self, user: _UserIndex, movie: _MovieIndex) -> int | None:
        """Retrieve the actual rating if it exists, otherwise return None."""
        value = self.y[movie, user].item()
        if 0 <= value <= 5:
            return int(value)
        assert value == -1
        return None

    def _calculate_prediction(self, user: _UserIndex, movie: _MovieIndex) -> float:
        """Calculate the predicted rating for a user and movie."""
        # Extract user parameters and movie features
        user_bias = self.p[user, 0]  # Bias term p_0
        user_features = self.p[user, 1:]  # Parameters p_1, p_2, ...
        movie_features = self.x[movie, :]  # Movie features x_1, x_2, ...

        # Calculate prediction
        prediction = user_bias + np.dot(user_features, movie_features)
        return prediction

    def __call__(self, user_id: UserId, movie_id: MovieId) -> int:
        """Generate a prediction for a given user and movie id."""
        # Convert UserId and MovieId to indices
        if user_id not in self.map_user or movie_id not in self.map_movie:
            raise ValueError(f"Invalid UserId {user_id} or MovieId {movie_id}.")
        user_idx = self.map_user[user_id]
        movie_idx = self.map_movie[movie_id]

        if (existing_rating := self._get_actual_rating(user_idx, movie_idx)) is not None:
            return existing_rating

        prediction = self._calculate_prediction(user_idx, movie_idx)
        prediction = np.clip(prediction, 0, 5)
        prediction = int(np.round(prediction, 0))
        assert 0 <= prediction <= 5
        return prediction
