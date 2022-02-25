#!/usr/bin/env python

from __future__ import absolute_import, division

import logging

import numpy

from .Algorithm import Algorithm

logger = logging.getLogger(__name__)


class Weighted_Sum(Algorithm):
    """weighted sum (default: mean)"""

    def __init__(self, weights=None, **kwargs):
        super(Weighted_Sum, self).__init__(classifier=self, **kwargs)
        self.weights = weights
        self.str["weights"] = weights

    def fit(self, X, y):
        pass

    def decision_function(self, scores):
        if self.weights is None:
            return numpy.mean(scores, axis=1)
        else:
            return numpy.sum(scores * self.weights, axis=1)
