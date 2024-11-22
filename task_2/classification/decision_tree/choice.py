from . import Movie


class Choice:
    """
    Class representing a single choice performed by the decision tree
    """

    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def test(self, movie: Movie) -> bool:
        feature = movie[self.feature_name]
        # todo: Implement testing
