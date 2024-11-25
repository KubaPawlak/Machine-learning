from abc import ABC, abstractmethod

from . import Movie


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


class CaregoricalChoice(Choice):
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
        return f"CaregoricalChoice({self.feature_name}, {self.value})"


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
