#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import bob.learn.mlp
import bob.core.random
import bob.io.base

from .AlgorithmBob import AlgorithmBob
from .mlp_train_helper import MLPTrainer

import logging

logger = logging.getLogger(__name__)


class MLP(AlgorithmBob):
    """This MLP is implemented using the bob tools.
    The preprocessors used with this class should be pickleable.
    """

    def __init__(self,
                 n_systems=2,
                 hidden_layers=(5,),
                 seed=None,
                 machine=None,
                 trainer=None,
                 batch_size=1,
                 epoch=1,
                 max_iter=1000,
                 no_improvements=0,
                 valley_condition=1,
                 **kwargs):
        super(MLP, self).__init__(classifier=self, **kwargs)
        self.mlp_shape = [n_systems] + list(hidden_layers) + [1]
        self.seed = seed
        self.machine = machine
        self.trainer = trainer
        self._mlp_trainer_args = {
            'batch_size': batch_size,
            'epoch': epoch,
            'max_iter': max_iter,
            'no_improvements': no_improvements,
            'valley_condition': valley_condition,
        }
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
        self.str.update({
            'seed': self.seed,
            'mlp_shape': self.mlp_shape,
            'machine': str(self.machine),
            'trainer': str(type(self.trainer))})

    def prepare_train(self, train, devel):
        (negatives, positives) = train
        n_systems = negatives.shape[1]
        if n_systems != self.mlp_shape[0]:
            logger.warn(
                'Reinitializing the MLP machine with the shape of {} to {} to '
                'match the input size.'.format(self.mlp_shape,
                                               [n_systems] +
                                               self.mlp_shape[1:]))
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
            **self._mlp_trainer_args)

    def train(self, train_neg, train_pos, devel_neg=None, devel_pos=None):
        if devel_neg is None:
            devel_neg = train_neg
        if devel_pos is None:
            devel_pos = train_pos
        self.prepare_train((train_neg, train_pos), (devel_neg, devel_pos))
        self.machine, self.analyzer = self.train_helper()

    def decision_function(self, scores):
        scores = self.machine(scores)
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores.ravel()
        return scores
