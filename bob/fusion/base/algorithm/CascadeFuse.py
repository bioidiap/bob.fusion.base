#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import numpy

from .Algorithm import Algorithm

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


class CascadeFuse(Algorithm):
  """weighted sum (default: mean)"""

  def __init__(self, *args, **kwargs):
    super(CascadeFuse, self).__init__(
      classifier=self,
      *args, **kwargs)

  def fit(self, X, y):
    # no training for this approach
    pass

  def decision_function(self, scores):
    # we assume that scores in 1st column are used to decide outcome
    return [score_set[0] if score_set[0] < 0.2 else score_set[1] for score_set in scores]
    # return [score_set[0] if score_set[0] == numpy.finfo(numpy.float16).min else score_set[1] for score_set in scores]
