#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

from ..tools import grouping
import numpy as np
import pickle

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


class Algorithm(object):
  """docstring for Algorithm"""

  def __init__(self,
               scores=None,
               performs_training=False,
               trainer_scores=None,
               has_closed_form_solution=False,
               preprocessors=None,
               *args,
               **kwargs
               ):
    """

  kwargs : ``key=value`` pairs
    A list of keyword arguments to be written in the
      :py:meth:`__str__` function.

"""
    super(Algorithm, self).__init__()
    self.scores = scores
    self.performs_training = performs_training
    self.trainer_scores = trainer_scores
    self.has_closed_form_solution = has_closed_form_solution
    self.preprocessors = preprocessors
    self._kwargs = kwargs
    self._kwargs['preprocessors'] = preprocessors

  def preprocess(self, scores):
    if self.preprocessors is not None:
      for i, (preprocessor, trained) in enumerate(self.preprocessors):
        if not trained:
          train_scores = np.vstack(self.trainer_scores)
          preprocessor.fit(train_scores)
          self.preprocessors[i] = (preprocessor, True)
        scores = self.preprocessor.transform(scores)
    return scores

  def train(self):
    negatives, positives = self.trainer_scores
    train_scores = np.vstack(self.trainer_scores)
    train_scores = self.preprocess(train_scores)
    neg_len = negatives.shape[0]
    y = np.zeros((train_scores.shape[0],), dtype='bool')
    y[neg_len:] = True
    self.fit(train_scores, y)

  def __call__(self):
    self.scores = self.preprocess(self.scores)
    return self.decision_function(self.scores)

  def plot_boundary_decision(self, score_labels, threshold,
                             label_system1='',
                             label_system2='',
                             thres_system1=None,
                             thres_system2=None,
                             do_grouping=False,
                             resolution=100,
                             x_pad=0.5,
                             y_pad=0.5,
                             alpha=0.75,
                             legends=None,
                             i1=0,
                             i2=1,
                             **kwargs
                             ):
    '''
    Plots the boundary decision of the Algorithm

    @param score_labels numpy.array A (self.scores.shape[0]) array containing
                                    the true labels of self.scores.

    @param threshold    float       threshold of the decision boundary
    '''
    if legends is None:
      legends = ['Impostor', 'Genuine']

    if self.scores.shape[1] > 2:
      raise NotImplementedError(
        "Currently plotting the decision boundary for more than two systems "
        "is not supported.")

    import matplotlib.pyplot as plt
    plt.gca()  # this is necessary for subplots to work.

    X = self.scores[:, [i1, i2]]
    Y = score_labels
    x_min, x_max = X[:, i1].min() - x_pad, X[:, i1].max() + x_pad
    y_min, y_max = X[:, i2].min() - y_pad, X[:, i2].max() + y_pad
    h1 = abs(x_max - x_min) / resolution
    h2 = abs(y_max - y_min) / resolution
    if self.has_closed_form_solution and self.scores.shape[1] == 2:
      x1 = np.arange(x_min, x_max, h1)
      x2 = self.closed_form(x1, threshold)
      plt.plot(x1, x2, cmap=plt.cm.viridis)
    else:
      xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h1), np.arange(y_min, y_max, h2))
      scores = self.scores
      self.scores = np.c_[xx.ravel(), yy.ravel()]
      Z = (self() > threshold).reshape(xx.shape)
      self.scores = scores

      contourf = plt.contour(xx, yy, Z, 1, alpha=1, cmap=plt.cm.viridis)

    if do_grouping:
      positives, negatives = X[Y], X[np.logical_not(Y)]
      negatives, positives = grouping(negatives, positives, **kwargs)
      X = np.concatenate((negatives, positives), axis=0)
      Y = np.concatenate(
        (np.zeros(negatives.shape[0], dtype=np.bool8),
         np.ones(positives.shape[0], dtype=np.bool8)),
        axis=0)

    plt.scatter(
      X[:, 0], X[:, 1], c=Y, alpha=alpha, cmap=plt.cm.viridis)
    # plt.legend(legends)

    if thres_system1 is not None:
      plt.axvline(thres_system1, color='red')
      plt.axhline(thres_system2, color='red')

    return contourf

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
    with open(model_file, "wb") as f:
      pickle.dump(self, f)

  def load(self, model_file):
    with open(model_file, "rb") as f:
      return pickle.load(f)
