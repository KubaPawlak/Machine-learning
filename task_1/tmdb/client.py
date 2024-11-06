import logging
import json
from os import getenv

import pandas as pd
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

    def extract_features(self, movie_data: dict) -> dict:
        features = {
            'genre': ';'.join([genre['name'] for genre in movie_data.get('genres', [])]),
            'rating': movie_data.get('vote_average', 0),
            'director': ';'.join([crew_member['name'] for crew_member in movie_data.get('credits', {}).get('crew', []) if crew_member['job'] == 'Director']),
            'cast': ';'.join([cast_member['name'] for cast_member in movie_data.get('credits', {}).get('cast', [])]),
            'release_date': movie_data.get('release_date', ''),
            'budget': movie_data.get('budget', 0),
            'revenue': movie_data.get('revenue', 0),
            'runtime': movie_data.get('runtime', 0),
            'language': ';'.join([language['iso_639_1'] for language in movie_data.get('spoken_languages', [])]),
            'country_of_origin': ';'.join([country['iso_3166_1'] for country in movie_data.get('production_countries', [])])
        }
        return features

def create_feature_dataframe(movie_ids, client):
    data = []
    for movie_id in movie_ids:
        movie_data = client.fetch_movie_data(movie_id)
        features = client.extract_features(movie_data)
        features['movie_id'] = movie_id
        data.append(features)
    df = pd.DataFrame(data)
    df.set_index('movie_id', inplace=True)
    return df

