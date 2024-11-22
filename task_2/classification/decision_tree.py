from collections import Counter
import numpy as np

type Movie = dict[str, int | float | list[int] | list[str]]

# Decision Tree Classifier
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini_impurity(self, y):
        """Calculate Gini Impurity."""
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)

    def _split_dataset(self, X, y, feature_index, threshold):
        """Split dataset based on feature and threshold."""
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        best_feature, best_threshold = None, None
        best_impurity = float("inf")
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                _, _, y_left, y_right = self._split_dataset(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Weighted impurity
                left_weight = len(y_left) / n_samples
                right_weight = len(y_right) / n_samples
                impurity = (
                    left_weight * self._gini_impurity(y_left)
                    + right_weight * self._gini_impurity(y_right)
                )

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree."""
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or depth == self.max_depth:
            return Counter(y).most_common(1)[0][0]  # Return the most common label

        feature, threshold = self._find_best_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]  # Return the most common label

        left_X, right_X, left_y, right_y = self._split_dataset(X, y, feature, threshold)
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(left_X, left_y, depth + 1),
            "right": self._build_tree(right_X, right_y, depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _predict(self, x, tree):
        """Traverse the tree for a prediction."""
        if not isinstance(tree, dict):
            return tree

        feature, threshold = tree["feature"], tree["threshold"]
        if x[feature] <= threshold:
            return self._predict(x, tree["left"])
        else:
            return self._predict(x, tree["right"])

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])
