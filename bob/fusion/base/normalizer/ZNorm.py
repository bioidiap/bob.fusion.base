#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import numpy

from .Normalizer import Normalizer

import logging
logger = logging.getLogger("bob.fusion.base")


class ZNorm(Normalizer):
  """the ZNorm score normalization"""

  def __init__(self,
               *args,
               **kwargs
               ):
    super(ZNorm, self).__init__(performs_training=True)

  def train(self, scores):
    super(ZNorm, self).train(scores)
    self.avg = numpy.average(scores, axis=0)
    self.std = numpy.std(scores, axis=0)

  def __call__(self, scores):
    scores = super(ZNorm, self).__call__(scores)
    return (scores - self.avg) / self.std
