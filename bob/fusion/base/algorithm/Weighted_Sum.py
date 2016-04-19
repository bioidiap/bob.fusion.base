#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import numpy

from .Algorithm import Algorithm

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


class Weighted_Sum(Algorithm):
  """weighted sum (default: mean)"""

  def __init__(self, weights=None, *args, **kwargs):
    super(Weighted_Sum, self).__init__(
      classifier=self,
      weights=weights,
      *args, **kwargs)
    self.weights = weights

  def fit(self, X, y):
    pass

  def decision_function(self, scores):
    if self.weights is None:
      return numpy.mean(scores, axis=1)
    else:
      return numpy.sum(scores * self.weights, axis=1)

  def closed_form(self, x1, y):
    if self.weights is None:
      return 2*y - x1
    else:
      w1 = self.weights[0]
      w2 = self.weights[1]
      x2 = (y - x1*w1)/w2
      return x2
