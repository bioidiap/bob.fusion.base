#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import bob.learn.linear

from .Algorithm import Algorithm

import logging
logger = logging.getLogger("bob.fusion.base")


class LLR(Algorithm):
  """docstring for LLR"""

  def __init__(self,
               *args, **kwargs):
    super(LLR, self).__init__(
        performs_training=True, *args, **kwargs)
    self.trainer = self.trainer if self.trainer else \
        bob.learn.linear.CGLogRegTrainer()

  def train(self):
    super(LLR, self).train()
    (negatives, positives) = self.trainer_scores
    # Trainning the LLR machine
    self.machine = self.trainer.train(negatives, positives)

  def __call__(self):
    super(LLR, self).__call__()
    # Applying the LLR in the input data
    return self.machine(self.scores).flatten()
