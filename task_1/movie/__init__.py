import json

import numpy as np


class Movie:

    def __init__(self,
                 movie_id,
                 title,
                 budget,
                 genres,
                 popularity,
                 release_year,
                 revenue,
                 runtime,
                 vote_average,
                 vote_count,
                 cast):
        self.movie_id: int = movie_id
        self.title: str = title
        self.budget: int = budget
        self.genres: list[int] = genres
        self.popularity: float = popularity
        self.release_year: int = release_year
        self.revenue: int = revenue
        self.runtime: int = runtime
        self.vote_average: float = vote_average
        self.vote_count: int = vote_count
        self.cast: list[str] = cast

    def __repr__(self) -> str:
        return json.dumps(self.__dict__)

    def to_feature_vector(self):
        # Normalize numerical similarity
        numerical_features = np.array([
            self.budget / 1e6,  # Scale budget to millions
            self.popularity,
            self.runtime / 60,  # Scale runtime to hours
            self.vote_average,
            self.vote_count / 1000,  # Scale vote_count
            self.release_year
        ])

        return numerical_features
