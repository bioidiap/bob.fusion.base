#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import numpy

from .Algorithm import Algorithm

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


class AND(Algorithm):
    """This algorithm fuses several systems with cascading."""

    def __init__(self, thresholds=None, *args, **kwargs):
        super(AND, self).__init__(
            classifier=self,
            thresholds=thresholds,
            *args, **kwargs)
        self.thresholds = thresholds

    def fit(self, X, y):
        pass

    def decision_function(self, scores):
        if self.thresholds is None:
            ValueError('No threshold was specified.')

        for i, th in enumerate(self.thresholds):
            mask = scores[:, i + 1] < th
            scores[mask, i + 1] = numpy.nan

        mask = numpy.sum(numpy.isnan(scores[:, 1:]), axis=1, dtype=bool)
        new_scores = numpy.array(scores[0])
        new_scores[mask] = numpy.finfo('float16').min
        return new_scores
