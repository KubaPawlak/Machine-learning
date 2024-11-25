from abc import ABC, abstractmethod
from typing import TypedDict

from . import Movie


class SplitResult(TypedDict):
    movies_passed: list[Movie]
    movies_failed: list[Movie]
    labels_passed: list[int]
    labels_failed: list[int]


class Choice(ABC):
    """
    Class representing a single choice performed by the decision tree
    """

    @abstractmethod
    def test(self, movie: Movie) -> bool:
        pass

    @abstractmethod
    def __str__(self):
        pass

    def split(self, movies: list[Movie], labels: list[int]) -> SplitResult:
        """Splits the list of movies into those which pass/fail the test respectively"""
        movies_passed = []
        movies_failed = []
        labels_passed = []
        labels_failed = []

        for movie, feature in zip(movies, labels):
            if self.test(movie):
                movies_passed.append(movie)
                labels_passed.append(feature)
            else:
                movies_failed.append(movie)
                labels_failed.append(feature)

        return {
            "movies_passed": movies_passed,
            "movies_failed": movies_failed,
            "labels_passed": labels_passed,
            "labels_failed": labels_failed,
        }


class ScalarChoice(Choice):
    """Represents choice made on scalar features (e.g. movie duration).
    Returns true if feature is greater or equal to the given value."""

    def __init__(self, feature_name: str, value: int | float):
        self.feature_name = feature_name
        self.value = value

    def test(self, movie: Movie) -> bool:
        return movie[self.feature_name] >= self.value

    def __str__(self):
        return f"{self.feature_name} >= {self.value}"

    def __repr__(self):
        return f"ScalarChoice({self.feature_name}, {self.value})"


class CategoricalChoice(Choice):
    """Choice made on categorical features (e.g. language).
    Returns true if the feature is the same as tested value."""

    def __init__(self, feature_name: str, value):
        self.feature_name = feature_name
        self.value = value

    def test(self, movie: Movie) -> bool:
        return movie[self.feature_name] == self.value

    def __str__(self):
        return f"{self.feature_name} = {self.value}"

    def __repr__(self):
        return f"CategoricalChoice({self.feature_name}, {self.value})"


class SetContainsChoice(Choice):
    """Choice made on multivalued features (e.g. cast).
    Checks if the tested value is contained in the feature."""

    def __init__(self, feature_name: str, value):
        self.feature_name = feature_name
        self.value = value

    def test(self, movie: Movie) -> bool:
        return self.value in set(movie[self.feature_name])

    def __str__(self):
        return f"{self.value} in {self.feature_name}"

    def __repr__(self):
        return f"SetContainsChoice({self.feature_name}, {self.value})"
