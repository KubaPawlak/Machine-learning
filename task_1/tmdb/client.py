import logging
import json
from os import getenv

import requests

from task_1.movie import Movie


def _map_response_to_movie(movie_data: dict) -> Movie:

    return Movie(
        title=movie_data['title'],
        runtime=movie_data['runtime'],
        budget=movie_data['budget'],
        revenue=movie_data['revenue'],
        genres=[genre['id'] for genre in movie_data['genres']],
        popularity=movie_data['popularity'],
        release_year=int(movie_data['release_date'][:4]),
        vote_average=movie_data['vote_average'],
        vote_count=movie_data['vote_count'],
        cast=[actor['name'] for actor in movie_data['credits']['cast']],
    )


class Client:
    def __init__(self, token: str | None = None):
        if token is None:
            token = getenv("TMDB_READ_ACCESS_TOKEN")
        if token is None:
            logging.error("TMDB_READ_ACCESS_TOKEN environment variable is not set")
            raise RuntimeError("TMDB_READ_ACCESS_TOKEN is not set")

        self.token: str = token

    def get_movie_raw(self, movie_id: int):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US&append_to_response=credits"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            logging.error(f"TMDB api returned response with status code {response.status_code}: {response.text}")
            raise RuntimeError(f"TMDB api invalid response: {response.text}")

        return json.loads(response.text)

    def get_movie(self, movie_id: int) -> Movie:
        movie_data = self.get_movie_raw(movie_id)
        return _map_response_to_movie(movie_data)
