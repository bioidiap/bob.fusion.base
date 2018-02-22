#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import time

import logging
import numpy

logger = logging.getLogger("bob.fusion.base")


class LLRCalibration(object):
    """LLR calibration """

    def __init__(self):
        self.trainers = None
        super(LLRCalibration, self).__init__()

    def fit_transform(self, scores, y=None):
        transformed_scores = scores
        if self.trainers is None and y is not None:
            self.trainers = []
            for i in range(scores.shape[1]):
                reshaped_scores = numpy.reshape(scores[:, i], (scores.shape[0], 1))
                # estimator = LogisticRegression(solver='lbfgs')
                # estimator.fit(reshaped_scores, y)
                trainer = CalibratedClassifierCV(LogisticRegression(solver='lbfgs'),  method='sigmoid')
                trainer.fit(reshaped_scores, y)
                transformed_column = trainer.predict_proba(reshaped_scores)[:, 1]
                # start = time.time()
                # trainer = LogisticRegression(C=1., solver='lbfgs')
                # trainer.fit(reshaped_scores, y)
                # end = time.time()
                # print("Time to fit: %f", float(end - start))
                # transformed_column = trainer.predict_proba(reshaped_scores)[:, 1]
                # predicted = trainer.decision_function(reshaped_scores)
                # transformed_column = (predicted - predicted.min()) / (predicted.max() - predicted.min())
                self.trainers.append(trainer)
                transformed_scores[:, i] = numpy.reshape(transformed_column, scores.shape[0])

        return transformed_scores

    def transform(self, scores):
        transformed_scores = scores
        assert len(self.trainers) == scores.shape[1]
        for i in range(len(self.trainers)):
            reshaped_scores = numpy.reshape(scores[:, i], (scores.shape[0], 1))
            # transformed_column = self.trainers[i].decision_function(reshaped_scores)
            transformed_column = self.trainers[i].predict_proba(reshaped_scores)[:, 1]
            # transformed_column = self.trainers[i].transform(reshaped_scores)
            # transformed_column = (predicted - predicted.min()) / (predicted.max() - predicted.min())
            transformed_scores[:, i] = numpy.reshape(transformed_column, scores.shape[0])
        return transformed_scores
