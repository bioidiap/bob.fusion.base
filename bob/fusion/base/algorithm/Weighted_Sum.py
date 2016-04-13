#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import numpy

from .Algorithm import Algorithm

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


class Weighted_Sum(Algorithm):
  """docstring for Weighted_Sum weighted sum (default: mean)"""

  def __init__(self, weights=None, *args, **kwargs):
    super(Weighted_Sum, self).__init__(
      performs_training=False, weights=weights,
      has_closed_form_solution=True, *args, **kwargs)
    self.weights = weights

  def decision_function(self, scores):
    if self.weights is None:
      return numpy.mean(scores, axis=1)
    else:
      return numpy.sum(scores * self.weights, axis=1)

  def closed_form(self, x1, y):
    return 2*y - x1
