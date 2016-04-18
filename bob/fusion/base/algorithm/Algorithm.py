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
               preprocessors=None,
               classifier=None,
               *args,
               **kwargs
               ):
    """

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
    (negatives, positives) = train
    train_scores = np.vstack((negatives, positives))
    neg_len = negatives.shape[0]
    y = np.zeros((train_scores.shape[0],), dtype='bool')
    y[neg_len:] = True
    self.classifier.fit(train_scores, y)

  def fuse(self, scores):
    if hasattr(self, 'classifier'):
      return self.classifier.decision_function(scores)
    else:
      return self.decision_function(scores)

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

  def plot_boundary_decision(self, scores, score_labels, threshold,
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

    @param score_labels numpy.array A (scores.shape[0]) array containing
                                    the true labels of scores.

    @param threshold    float       threshold of the decision boundary
    '''
    if legends is None:
      legends = ['Impostor', 'Genuine']
    markers = ['x', 'o']

    if scores.shape[1] > 2:
      raise NotImplementedError(
        "Currently plotting the decision boundary for more than two systems "
        "is not supported.")

    import matplotlib.pyplot as plt
    plt.gca()  # this is necessary for subplots to work.

    X = scores[:, [i1, i2]]
    Y = score_labels
    x_min, x_max = X[:, i1].min() - x_pad, X[:, i1].max() + x_pad
    y_min, y_max = X[:, i2].min() - y_pad, X[:, i2].max() + y_pad
    xx, yy = np.meshgrid(
      np.linspace(x_min, x_max, resolution),
      np.linspace(y_min, y_max, resolution))
    temp = np.c_[xx.ravel(), yy.ravel()]
    temp = self.preprocess(temp)
    Z = (self.fuse(temp) > threshold).reshape(xx.shape)

    contourf = plt.contour(xx, yy, Z, 1, alpha=1, cmap=plt.cm.viridis)

    if do_grouping:
      negatives, positives = X[np.logical_not(Y)], X[Y]
      negatives, positives = grouping(negatives, positives, **kwargs)
      X = np.concatenate((negatives, positives), axis=0)
      Y = np.concatenate(
        (np.zeros(negatives.shape[0], dtype=np.bool8),
         np.ones(positives.shape[0], dtype=np.bool8)),
        axis=0)

    negatives, positives = X[np.logical_not(Y)], X[Y]
    colors = plt.cm.viridis(np.linspace(0, 1, 2))
    for i, X in enumerate((negatives, positives)):
      plt.scatter(
        X[:, 0], X[:, 1], marker=markers[i], alpha=alpha,
        c=colors[i], label=legends[i])
    plt.legend()

    if thres_system1 is not None:
      plt.axvline(thres_system1, color='red')
      plt.axhline(thres_system2, color='red')

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    return contourf
