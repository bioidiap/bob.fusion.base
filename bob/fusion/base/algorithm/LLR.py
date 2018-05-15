#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import bob.learn.linear

from .AlgorithmBob import AlgorithmBob

import logging
logger = logging.getLogger("bob.fusion.base")


class LLR(AlgorithmBob):
    """LLR Score fusion using Bob"""

    def __init__(self,
                 trainer=None,
                 **kwargs):
        self.trainer = trainer if trainer else \
            bob.learn.linear.CGLogRegTrainer()
        # this is needed to be able to load the machine
        self.machine = bob.learn.linear.Machine()
        super(LLR, self).__init__(classifier=self, **kwargs)
        self.str['trainer'] = str(type(self.trainer))

    def train(self, train_neg, train_pos, devel_neg=None, devel_pos=None):
        # Trainning the LLR machine
        self.machine = self.trainer.train(train_neg, train_pos)

    def decision_function(self, scores):
        scores = self.machine(scores)
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores.ravel()
        return scores
