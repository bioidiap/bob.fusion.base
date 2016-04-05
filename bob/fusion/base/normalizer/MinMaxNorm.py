#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import numpy

from .Normalizer import Normalizer

import logging
logger = logging.getLogger("bob.fusion.base")


class MinMaxNorm(Normalizer):
  """
  the MinMaxNorm score normalization
  Normalize the score in an specific interval

  @param lowBound The lower bound
  @param upperBound The upper bound
  """

  def __init__(self,
               lowerBound=-1,
               upperBound=1,
               *args,
               **kwargs
               ):
    super(MinMaxNorm, self).__init__(performs_training=True)
    self.lowerBound = lowerBound
    self.upperBound = upperBound

  def train(self, scores):
    super(MinMaxNorm, self).train(scores)
    self.mins = numpy.min(scores, axis=0)
    self.maxs = numpy.max(scores, axis=0)

  def __call__(self, scores):
    scores = super(MinMaxNorm, self).__call__(scores)
    denom = self.maxs - self.mins
    normalizedScores = (self.upperBound - self.lowerBound) * \
        (scores - self.mins) / denom + self.lowerBound

    return normalizedScores
