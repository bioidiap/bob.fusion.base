#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import bob.learn.mlp
import bob.core.random
import bob.io.base
import pickle

from .Algorithm import Algorithm
from .mlp_train_helper import MLPTrainer

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


class MLP(Algorithm):
  """This MLP is implemented using the bob tools.
  The preprocessors used with this class should be pickleable.
  """

  def __init__(self,
               n_systems=2,
               hidden_layers=None,
               seed=None,
               machine=None,
               trainer=None,
               *args, **kwargs):
    super(MLP, self).__init__(
        classifier=self,
        *args, **kwargs)
    if hidden_layers is None:
      hidden_layers = [3]
    self.mlp_shape = [n_systems] + hidden_layers + [1]
    self.seed = seed
    self.machine = machine
    self.trainer = trainer
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
    self._kwargs = {
      'seed': self.seed,
      'mlp_shape': self.mlp_shape,
      'machine': self.machine,
      'train': self.train,
    }

  def prepare_train(self, train, devel):
    (negatives, positives) = train
    n_systems = negatives.shape[1]
    if n_systems != self.mlp_shape[0]:
      logger.warn(
        'Reinitializing the MLP machine with the shape of {} to {} to match th'
        'e input size.'.format(self.mlp_shape, [n_systems]+self.mlp_shape[1:]))
      self.mlp_shape = [n_systems] + self.mlp_shape[1:]
      self.n_systems = n_systems
      self.hidden_layers = self.mlp_shape[1:-1]
      self.initialize(force=True)
    self.train_helper = MLPTrainer(
        train=train[::-1],
        devel=devel[::-1],
        mlp_shape=self.mlp_shape,
        machine=self.machine,
        trainer=self.trainer,
        **self._my_kwargs)

  def train(self, train, devel=None):
    if devel is None:
      devel = train
    self.prepare_train(train, devel)
    self.machine, self.analyzer = self.train_helper()

  def decision_function(self, scores):
    scores = self.machine(scores)
    if scores.ndim == 2 and scores.shape[1] == 1:
      scores = scores.ravel()
    return scores

  def _get_hdf5_file(self, model_file):
    return model_file[:-3] + 'hdf5'

  def save(self, model_file):
    d5 = bob.io.base.HDF5File(self._get_hdf5_file(model_file), "w")
    try:
      self.machine.save(d5)
    finally:
      d5.close()

    # dump preprocessors in a pickle file because
    # we don't know how they look like
    with open(model_file, 'wb') as f:
      pickle.dump(self.preprocessors, f)

  def load(self, model_file):
    d5 = bob.io.base.HDF5File(self._get_hdf5_file(model_file))
    try:
      self.machine.load(d5)
    finally:
      d5.close()

    # load preprocessors
    with open(model_file, "rb") as f:
      self.preprocessors = pickle.load(f)

    return self
