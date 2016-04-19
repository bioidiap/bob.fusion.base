#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import numpy as np
import pickle

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


class Algorithm(object):
  """A class to be used in score fusion"""

  def __init__(self,
               preprocessors=None,
               classifier=None,
               *args,
               **kwargs
               ):
    """
  preprocessors: A list of preprocessors that follow the API of
    :py:meth:`sklearn.preprocessing.StandardScaler`. Especially `fit_transform`
    and `transform` must be implemented.

  classifier: An instance of a class that implements `fit(X[, y])` and
    `decision_function(X)` like:
    :py:meth:`sklearn.linear_model.LogisticRegression`

  kwargs : ``key=value`` pairs
    A list of keyword arguments to be written in the
      :py:meth:`__str__` function.

"""
    super(Algorithm, self).__init__()
    self.classifier = classifier
    self.preprocessors = preprocessors
    self._kwargs = kwargs
    self._kwargs['preprocessors'] = preprocessors

  def train_preprocessors(self, X):
    if self.preprocessors is not None:
      for preprocessor in self.preprocessors:
        X = preprocessor.fit_transform(X)

  def preprocess(self, scores):
    if self.preprocessors is not None:
      for preprocessor in self.preprocessors:
        scores = preprocessor.transform(scores)
    return scores

  def train(self, train, devel=None):
    """If you use development data for training you need to override this
    method.
    train: A :py:meth:`tuple` of length 2 containing
      the negatives and positives. negatives and positives should be
      numpy.ndarray with the shape of (n_samples, n_systems).
    devel: same as train but used for development (validation)
    """
    (negatives, positives) = train
    train_scores = np.vstack((negatives, positives))
    neg_len = negatives.shape[0]
    y = np.zeros((train_scores.shape[0],), dtype='bool')
    y[neg_len:] = True
    self.classifier.fit(train_scores, y)

  def fuse(self, scores):
    """
    scores: A numpy.ndarray with the shape of (n_samples, n_systems).
    """
    return self.classifier.decision_function(scores)

  def __str__(self):
    """__str__() -> info

    This function returns all parameters of this class (and its derived class).

    **Returns:**

    info : str
      A string containing the full information of all parameters of this
        (and the derived) class.
    """
    return "%s(%s)" % (str(self.__class__), ", ".join(
      ["%s=%s" % (key, value) for key, value in
       self._kwargs.items() if value is not None]))

  def save(self, model_file):
    """If your class cannot be pickled, you need to override this method."""
    with open(model_file, "wb") as f:
      pickle.dump(self, f)

  def load(self, model_file):
    """If your class cannot be pickled, you need to override this method."""
    with open(model_file, "rb") as f:
      return pickle.load(f)
