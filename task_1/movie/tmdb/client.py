import atexit
import json
import logging
from os import getenv
from pathlib import Path
from pickle import dump, load

import requests

from data.movie import movies
from task_1.movie import Movie

_CACHE_PATH = Path(__file__).parent / 'response_cache.pkl'


def _load_response_cache() -> dict[int, str]:
    try:
        with open(_CACHE_PATH, 'rb') as file:
            return load(file)
    except FileNotFoundError:
        return {}


def _save_response_cache():
    with open(_CACHE_PATH, 'wb') as file:
        # noinspection PyTypeChecker
        dump(_response_cache, file)


_response_cache: dict[int, str] = _load_response_cache()
atexit.register(_save_response_cache)


def _map_response_to_movie(movie_id: int, movie_data: dict) -> Movie:
    return Movie(
        movie_id=movie_id,
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
        director=[member['name'] for member in movie_data['credits']['crew'] if member['job'] == 'Director'],
    )


def _lookup_tmdb_id(movie_id: int):
    return int(movies[movies['ID'] == movie_id]['TMDBID'].iloc[0])


class Client:
    def __init__(self, token: str | None = None):
        if token is None:
            token = getenv("TMDB_READ_ACCESS_TOKEN")
        if token is None:
            logging.warning("TMDB_READ_ACCESS_TOKEN environment variable is not set. Will fail if movie is not cached.")

        self.token: str | None = token

    def _call_api(self, movie_id: int) -> str:
        if self.token is None:
            logging.error("TMDB_READ_ACCESS_TOKEN environment variable is not set")
            raise RuntimeError("TMDB_READ_ACCESS_TOKEN is not set")

        tmdb_id = _lookup_tmdb_id(movie_id)

        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?language=en-US&append_to_response=credits"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            logging.error(
                f"TMDB api returned response with status code {response.status_code}, movie_id={tmdb_id}({movie_id}): {response.text}")
            raise RuntimeError(f"TMDB api invalid response: {response.text}")

        return response.text

    def get_movie_raw(self, movie_id: int):
        if movie_id in _response_cache:
            response = _response_cache[movie_id]
        else:
            response = self._call_api(movie_id)
            _response_cache[movie_id] = response

        assert movie_id in _response_cache

        return json.loads(response)

    def get_movie(self, movie_id: int) -> Movie:
        movie_data = self.get_movie_raw(movie_id)
        return _map_response_to_movie(movie_id, movie_data)
