import numpy as np

from movie import Movie
from task_2.classification.decision_tree.choice import Choice, SplitResult, ScalarChoice, CategoricalChoice, \
    SetContainsChoice
from ._movie import MovieDict


# Decision Tree Classifier
def _gini_impurity(y):
    """Calculate Gini Impurity."""
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)


def _as_dict(movies: list[Movie] | list[MovieDict] | Movie | MovieDict):
    if isinstance(movies, list):
        if isinstance(movies[0], Movie):
            return list(map(lambda m: m.__dict__, movies))
        return movies
    else:
        movie = movies
        if isinstance(movie, Movie):
            return movie.__dict__
        return movie


class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.is_fitted: bool = False
        # if choice node:
        self.choice: Choice | None = None
        self.child_success: DecisionTree | None = None
        self.child_fail: DecisionTree | None = None
        # if leaf node:
        self.leaf_value: int | None = None

    def ensure_valid(self) -> None:
        """Ensures that the tree has valid structure.
        The tree must be either a leaf node, or a choice node"""
        if self.leaf_value is None:
            # must be a choice node
            assert self.choice is not None, "Choice cannot be None in non-leaf node"
            assert self.child_success is not None and self.child_fail is not None, "Choice node must have children"
        else:
            assert self.choice is None, "Choice must be None in leaf node"
            assert self.child_success is None and self.child_fail is None, "Leaf node cannot have children"

    @staticmethod
    def _find_best_choice(movies: list[MovieDict], labels: list[int]) -> Choice:
        possible_choices: list[Choice] = []

        for feature_name in movies[0].keys():
            if feature_name == 'title':
                # this feature uniquely identifies each movie, it is useless for generalization
                continue
            unique_values: set[int | float | str] = set()
            isMultivalued = False
            for movie in movies:
                feature = movie[feature_name]
                if isinstance(feature, list):
                    isMultivalued = True
                    unique_values = {*unique_values, *feature}
                else:
                    unique_values.add(feature)

            if isMultivalued:
                for value in unique_values:
                    possible_choices.append(SetContainsChoice(feature_name, value))
            elif isinstance(next(iter(unique_values)), str):
                # feature values is string, therefore is categorical
                for value in unique_values:
                    possible_choices.append(CategoricalChoice(feature_name, value))
            else:
                for threshold in unique_values:
                    possible_choices.append(ScalarChoice(feature_name, threshold))

        def rate_split(split: SplitResult) -> float:
            """Calculates the rating of how good a split is. Higher value = better split."""
            total_count = len(split['labels_passed']) + len(split['labels_failed'])

            if total_count == 0:
                return 0  # Avoid division by zero

            gini_passed = _gini_impurity(split['labels_passed'])
            gini_failed = _gini_impurity(split['labels_failed'])

            # Calculate weighted Gini impurity for the split
            weighted_impurity = (
                    (len(split['labels_passed']) / total_count) * gini_passed +
                    (len(split['labels_failed']) / total_count) * gini_failed
            )

            # Return 1 - weighted impurity to ensure higher values indicate better splits
            return 1 - weighted_impurity

        def rate_choice(c: Choice) -> float:
            split = c.split(movies, labels)
            # print(f"Evaluating choice: {c}, Split result: {split}")
            return rate_split(split)

        # Evaluate each possible choice and find the best one
        choice_ratings = list(map(rate_choice, possible_choices))
        best_choice_idx = np.argmax(choice_ratings)
        # print(f"Best choice: {possible_choices[best_choice_idx]}")
        return possible_choices[best_choice_idx]

    def fit(self, movies: list[MovieDict] | list[Movie], labels: list[int]) -> None:
        assert len(movies) == len(labels), "Movies and labels must have same length"
        assert len(movies) > 0, "Attempt to fit to an empty movie list"
        movies: list[MovieDict] = _as_dict(movies)
        assert self.max_depth >= 1, "max_depth must be at least 1"

        if len(np.unique(labels)) == 1:
            # All movies have the same label
            self.leaf_value = int(labels[0])
        elif self.max_depth == 1:
            # Must be a leaf node because we have run out of depth
            most_frequent = np.argmax(np.bincount(labels)).item()
            self.leaf_value = most_frequent
        else:
            # Attempt to find the best choice for splitting
            self.choice = DecisionTree._find_best_choice(movies, labels)
            split_dataset = self.choice.split(movies, labels)

            # Check if the split is valid
            assert len(split_dataset['movies_passed']) > 0 and len(
                split_dataset['movies_failed']) > 0, "Split not splitting"

            # Create and fit child trees
            self.child_success = DecisionTree(max_depth=self.max_depth - 1)
            self.child_fail = DecisionTree(max_depth=self.max_depth - 1)
            self.child_success.fit(split_dataset['movies_passed'], split_dataset['labels_passed'])
            self.child_fail.fit(split_dataset['movies_failed'], split_dataset['labels_failed'])

        self.ensure_valid()
        self.is_fitted = True

    def _predict_single(self, movie: Movie | MovieDict) -> int:
        movie: MovieDict = _as_dict(movie)

        if self.leaf_value is not None:
            return self.leaf_value

        passed_check = self.choice.test(movie)
        if passed_check:
            return self.child_success._predict_single(movie)
        else:
            return self.child_fail._predict_single(movie)

    def predict(self, movies: list[Movie]) -> list[int]:
        return list(map(self._predict_single, movies))
