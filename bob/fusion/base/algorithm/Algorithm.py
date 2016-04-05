#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

from ..utils import grouping
import numpy as np


class Algorithm(object):
  """docstring for Algorithm"""

  def __init__(self,
               scores=None,
               normalizer=None,
               performs_training=False,
               trainer_scores=None,
               trainer=None,
               machine=None,
               *args,
               **kwargs
               ):
    super(Algorithm, self).__init__()
    self.scores = scores
    self.performs_training = performs_training
    self.trainer_scores = trainer_scores
    self.trainer = trainer
    self.machine = machine
    self.normalizer = normalizer

  def normalize(self, scores):
    if self.normalizer is None:
      return scores
    else:
      if not self.normalizer.trained:
        train_scores = np.vstack(self.trainer_scores)
        self.normalizer.train(train_scores)
      return self.normalizer(scores)

  def train(self):
    negatives, positives = self.trainer_scores
    negatives = self.normalize(negatives)
    positives = self.normalize(positives)
    self.trainer_scores = (negatives, positives)

  def __call__(self):
    self.scores = self.normalize(self.scores)

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
    import matplotlib.pyplot as plt
    plt.gca()  # this is necessary for subplots to work.

    X = self.scores
    Y = score_labels
    x_min, x_max = X[:, 0].min() - x_pad, X[:, 0].max() + x_pad
    y_min, y_max = X[:, 1].min() - y_pad, X[:, 1].max() + y_pad
    h1 = abs(x_max - x_min) / resolution
    h2 = abs(y_max - y_min) / resolution
    xx, yy = np.meshgrid(
      np.arange(x_min, x_max, h1), np.arange(y_min, y_max, h2))
    self.scores = np.c_[xx.ravel(), yy.ravel()]
    Z = (self() > threshold).reshape(xx.shape)
    self.scores = X

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
