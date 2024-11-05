import logging
import json
from os import getenv

import requests


class Client:
    def __init__(self, token: str | None = None):
        if token is None:
            token = getenv("TMDB_READ_ACCESS_TOKEN")
        if token is None:
            logging.error("TMDB_READ_ACCESS_TOKEN environment variable is not set")
            raise RuntimeError("TMDB_READ_ACCESS_TOKEN is not set")

        self.token: str = token

    def fetch_movie_data(self, movie_id: int):
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
