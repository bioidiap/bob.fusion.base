#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import bob.learn.linear
from sklearn.linear_model import LogisticRegression as LogisticRegression_SK

from .Algorithm import Algorithm

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


class LogisticRegression(Algorithm, LogisticRegression_SK):
  __doc__ = LogisticRegression_SK.__doc__

  def __init__(self,
               *args, **kwargs):
    Algorithm.__init__(
        self, performs_training=True,
        has_closed_form_solution=True, *args, **kwargs)
    sk_kwargs = {}
    for key, value in kwargs.items():
      if key in ['penalty', 'dual', 'tol', 'C', 'fit_intercept',
                 'intercept_scaling', 'class_weight',
                 'random_state', 'solver', 'max_iter',
                 'multi_class', 'verbose', 'warm_start', 'n_jobs']:
        sk_kwargs[key] = value

    LogisticRegression_SK.__init__(self, **sk_kwargs)

  def closed_form(self, x1, y):
    w1 = self.coef_[0]
    w2 = self.coef_[1]
    x2 = (y - self.intercept_ - x1*w1)/w2
    return x2
