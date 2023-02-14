#!/usr/bin/env python

from __future__ import absolute_import, division

import logging

from .Algorithm import Algorithm

logger = logging.getLogger(__name__)


class Empty(Algorithm):
    """Empty algorithm
    This algorithm does not change scores by itself and only applies the
    preprocessors."""

    def __init__(self, **kwargs):
        super(Empty, self).__init__(classifier=self, **kwargs)

    def fit(self, X, y):
        pass

    def decision_function(self, scores):
        return scores.flat
