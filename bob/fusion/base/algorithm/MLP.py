#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import bob.learn.mlp
import bob.core.random

import numpy

from .Algorithm import Algorithm

import logging
logger = logging.getLogger("bob.fusion.base")


class MLP(Algorithm):
  """docstring for MLP"""

  def __init__(self,
               mlp_shape=None,
               trainer_devel=None,
               seed=None,
               *args, **kwargs):
    super(MLP, self).__init__(
        performs_training=True, *args, **kwargs)
    if mlp_shape is not None:
      self.mlp_shape = mlp_shape
    elif self.scores is not None:
      self.mlp_shape = (numpy.asarray(self.scores).shape[1], 3, 1)
    else:
      self.mlp_shape = (2, 3, 1)
    self.machine = self.machine if self.machine else \
        bob.learn.mlp.Machine(self.mlp_shape)
    if seed is not None:
      self.rng = bob.core.random.mt19937(seed)
      self.machine.randomize(rng=self.rng)
    else:
      self.machine.randomize()
    self.trainer = self.trainer if self.trainer else \
        bob.learn.mlp.RProp(1, bob.learn.mlp.SquareError(
            self.machine.output_activation), machine=self.machine,
          train_biases=False)
    self.trainer_devel = trainer_devel if trainer_devel else \
        self.trainer_scores
    self.train_helper = bob.learn.mlp.MLPTrainer(
        train=self.trainer_scores[::-1],
        devel=self.trainer_devel[::-1],
        mlp_shape=self.mlp_shape,
        machine=self.machine,
        trainer=self.trainer,
        **kwargs)

  def train(self):
    super(MLP, self).train()
    self.machine, self.analyzer = self.train_helper()

  def __call__(self):
    super(MLP, self).__call__()
    return self.machine(self.scores).flatten()
