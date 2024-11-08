import json


class Movie:

    def __init__(self, title, budget, genres, popularity, release_year, revenue, runtime, vote_average, vote_count,
                 cast):
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
