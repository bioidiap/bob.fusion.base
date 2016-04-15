#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import bob.learn.mlp
import bob.core.random
import bob.io.base
import numpy

from .Algorithm import Algorithm
from .mlp_train_helper import MLPTrainer

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


class MLP(Algorithm):
  """This MLP is implemented using the bob tools
  It may change its API and functionality in the future.
  """

  def __init__(self,
               n_systems=2,
               hidden_layers=None,
               trainer_devel=None,
               seed=None,
               machine=None,
               trainer=None,
               *args, **kwargs):
    # chicken and egg :D call __init__ twice.
    super(MLP, self).__init__(performs_training=True, *args, **kwargs)
    if hidden_layers is None:
      hidden_layers = [3]
    if self.scores is not None:
      n_systems = numpy.asarray(self.scores).shape[1]
    self.mlp_shape = [n_systems] + hidden_layers + [1]
    super(MLP, self).__init__(
        performs_training=True, mlp_shape=self.mlp_shape, seed=seed,
        machine=str(machine), trainer=str(trainer),
        *args, **kwargs)
    self.seed = seed
    self.machine = machine
    self.trainer = trainer
    self.trainer_devel = trainer_devel if trainer_devel else \
        self.trainer_scores
    self._my_kwargs = kwargs
    self.initialize()

  def initialize(self, force=False):
    self.machine = self.machine if self.machine and not force else \
        bob.learn.mlp.Machine(self.mlp_shape)
    if self.seed is not None:
      self.rng = bob.core.random.mt19937(self.seed)
      self.machine.randomize(rng=self.rng)
    else:
      self.machine.randomize()
    self.trainer = self.trainer if self.trainer and not force else \
        bob.learn.mlp.RProp(1, bob.learn.mlp.SquareError(
            self.machine.output_activation), machine=self.machine,
          train_biases=False)

  def prepare_train(self):
    self.trainer_devel = self.trainer_devel if self.trainer_devel else \
        self.trainer_scores
    self.train_helper = MLPTrainer(
        train=self.trainer_scores[::-1],
        devel=self.trainer_devel[::-1],
        mlp_shape=self.mlp_shape,
        machine=self.machine,
        trainer=self.trainer,
        **self._my_kwargs)

  def fit(self, train_scores, y):
    n_systems = train_scores.shape[1]
    if n_systems != self.mlp_shape[0]:
      logger.warn(
        'Reinitializing the MLP machine with the shape of {} to {} to match th'
        'e input size.'.format(self.mlp_shape, [n_systems]+self.mlp_shape[1:]))
      self.mlp_shape = [n_systems] + self.mlp_shape[1:]
      self.n_systems = n_systems
      self.hidden_layers = self.mlp_shape[1:-1]
      self.initialize(force=True)
    self.trainer_scores = (train_scores[numpy.logical_not(y)], train_scores[y])
    self.prepare_train()
    self.machine, self.analyzer = self.train_helper()

  def decision_function(self, scores):
    scores = self.machine(scores)
    if scores.ndim == 2 and scores.shape[1] == 1:
      scores = scores.ravel()
    return scores

  def save(self, model_file):
    d5 = bob.io.base.HDF5File(model_file, "w")
    try:
      self.machine.save(d5)
    finally:
      d5.close()

  def load(self, model_file):
    d5 = bob.io.base.HDF5File(model_file)
    try:
      self.machine.load(d5)
    finally:
      d5.close()
