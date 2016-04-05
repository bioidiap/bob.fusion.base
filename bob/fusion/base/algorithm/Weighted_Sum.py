#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import numpy

from .Algorithm import Algorithm

import logging
logger = logging.getLogger("bob.fusion.base")


class Weighted_Sum(Algorithm):
  """docstring for Weighted_Sum weighted sum (default: mean)"""

  def __init__(self, weights=None, *args, **kwargs):
    super(Weighted_Sum, self).__init__(
      performs_training=False, *args, **kwargs)
    self.weights = weights

  def __call__(self):
    super(Weighted_Sum, self).__call__()
    if self.weights is None:
      return numpy.mean(self.scores, axis=1)
    else:
      return numpy.sum(self.scores * self.weights, axis=1)
