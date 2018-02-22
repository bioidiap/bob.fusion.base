#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

from .Algorithm import Algorithm
from scipy.stats.mstats import hmean
import numpy

import bob.core

logger = bob.core.log.setup("bob.fusion.base")


class HarmonicMean(Algorithm):
    """harmonic sum (default: mean)"""

    def __init__(self, *args, **kwargs):
        super(HarmonicMean, self).__init__(
            classifier=self,
            *args, **kwargs)

    def fit(self, X, y):
        pass

    def decision_function(self, scores):
        return hmean(numpy.square(scores), axis=1)
