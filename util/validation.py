import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from .submission import Model

type AccuracyScores = tuple[float, float]


class Validator:
    def __init__(self, model: Model):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _calculate_scores(actual: Iterable[int], predicted: Iterable[int]) -> AccuracyScores:
        correct = 0
        one_off = 0
        total = 0
        for (y_true, y_pred) in zip(actual, predicted):
            total += 1
            if y_true == y_pred:
                correct += 1
            if (y_true - y_pred) in [-1, 1]:
                one_off += 1

        return correct / total, one_off / total

    def _report_scores(self, fraction_correct: float, fraction_one_off: float):
        self.logger.info(f"Correct: {fraction_correct * 100:.2f}%")
        self.logger.info(f"One off: {fraction_one_off * 100:.2f}%")

    def train_set_accuracy(self) -> AccuracyScores:
        test: pd.DataFrame = self.model.train_set.copy()
        test['Rating'] = np.nan
        result = self.model.generate_submission(test)
        scores = self._calculate_scores(result['Rating'], result['Rating'])
        self._report_scores(*scores)
        return scores

    def _train_and_predict(self, train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
        with self.model.with_custom_train_set(train):
            test = val.copy()
            test['Rating'] = np.nan
            return self.model.generate_submission(test)


    def validation_set_accuracy(self, validation_set_fraction: float = 0.2) -> AccuracyScores:
        assert 0 < validation_set_fraction < 1
        train, val = train_test_split(self.model.train_set, test_size=validation_set_fraction)
        result = self._train_and_predict(train, val)
        scores = self._calculate_scores(val['Rating'], result['Rating'])
        self._report_scores(*scores)
        return scores

    def k_fold_cross_validation(self, k=5) -> AccuracyScores:
        k_fold = KFold(n_splits=k, shuffle=True)
        whole_set = self.model.train_set
        scores: list[AccuracyScores] = []
        for train_loc, val_loc in k_fold.split(whole_set):
            train = whole_set.iloc[train_loc]
            val = whole_set.iloc[val_loc]
            result = self._train_and_predict(train, val)
            current_score = self._calculate_scores(val['Rating'], result['Rating'])
            scores.append(current_score)

        aggregate_scores = np.array(scores).mean(axis=0)
        aggregate_scores: AccuracyScores = float(aggregate_scores[0]), float(aggregate_scores[1])
        self._report_scores(*aggregate_scores)
        return aggregate_scores
