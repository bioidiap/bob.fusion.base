#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import


class Normalizer(object):
  """docstring for Normalizer"""

  def __init__(self,
               performs_training=False,
               trained=False,
               *args,
               **kwargs
               ):
    super(Normalizer, self).__init__()
    self.performs_training = performs_training
    if not self.performs_training:
      trained = True
    self.trained = trained

  def train(self, scores):
    """
    Trains the Normalizer
    calls to this function changes the self.trained to True
    @param scores numpy.array of scores to be used for training
    """
    self.trained = True

  def __call__(self, scores):
    """
    Normalizes the scores
    @param scores numpy.array to be normalized
    @return numpy.array with the normalized scores.
    """
    return scores
