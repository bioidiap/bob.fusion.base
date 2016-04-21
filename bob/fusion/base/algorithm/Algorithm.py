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
    if classifier is not self:
      self._kwargs['classifier'] = classifier

  def train_preprocessors(self, X):
    """Train preprocessors in order.
    X: numpy.ndarray with the shape of (n_samples, n_systems)."""
    if self.preprocessors is not None:
      for preprocessor in self.preprocessors:
        X = preprocessor.fit_transform(X)

  def preprocess(self, scores):
    """
    scores: numpy.ndarray with the shape of (n_samples, n_systems).
    returns the transformed scores."""
    if self.preprocessors is not None:
      for preprocessor in self.preprocessors:
        scores = preprocessor.transform(scores)
    return scores

  def train(self, train_neg, train_pos, devel_neg=None, devel_pos=None):
    """If you use development data for training you need to override this
    method.

    train_neg: numpy.ndarray
      Negatives training data should be numpy.ndarray with the shape of
      (n_samples, n_systems).
    train_pos: numpy.ndarray
      Positives training data should be numpy.ndarray with the shape of
      (n_samples, n_systems).
    devel_neg, devel_pos: numpy.ndarray
      Same as ``train`` but used for development (validation).
    """
    train_scores = np.vstack((train_neg, train_pos))
    neg_len = train_neg.shape[0]
    y = np.zeros((train_scores.shape[0],), dtype='bool')
    y[neg_len:] = True
    self.classifier.fit(train_scores, y)

  def fuse(self, scores):
    """
    scores: numpy.ndarray
      A numpy.ndarray with the shape of (n_samples, n_systems).

    **Returns:**

    fused_score: numpy.ndarray
      The fused scores in shape of (n_samples,).
    """
    return self.classifier.decision_function(scores)

  def __str__(self):
    """Return all parameters of this class (and its derived class) in string.


    **Returns:**

    info: str
      A string containing the full information of all parameters of this
        (and the derived) class.
    """
    return "%s(%s)" % (str(self.__class__), ", ".join(
        ["%s=%s" % (key, value) for key, value in
         self._kwargs.items() if value is not None]))

  def save(self, model_file):
    """Save the instance of the algorithm.

    model_file: str
      A path to save the file. Please note that file objects
      are not accepted. The filename MUST end with ".pkl".
      Also, an algorithm may save itself in multiple files with different
      extensions such as model_file and model_file[:-3]+'hdf5'.
    """
    # support for bob machines
    if hasattr(self, "machine"):
      self.save_bob(model_file)
    else:
      with open(model_file, "wb") as f:
        pickle.dump(self, f)

  def load(self, model_file):
    """Load the algorithm the same way it was saved.
    A new instance will be returned.

    **Returns:**

    loaded_algorithm: Algorithm
      A new instance of the loaded algorithm.
    """
    with open(model_file, "rb") as f:
      temp = pickle.load(f)
    if isinstance(temp, Algorithm):
      return temp
    else:
      return self.load_bob(model_file)

  def _get_hdf5_file(self, model_file):
    return model_file[:-3] + 'hdf5'

  def save_bob(self, model_file):
    # dump preprocessors in a pickle file because
    # we don't know how they look like
    # saves the class to create it later.
    with open(model_file, 'wb') as f:
      pickle.dump(self.preprocessors, f)
      pickle.dump(type(self), f)
      # just for consistent string representation
      pickle.dump(self._kwargs, f)

    d5 = bob.io.base.HDF5File(self._get_hdf5_file(model_file), "w")
    try:
      self.machine.save(d5)
    finally:
      d5.close()

  def load_bob(self, model_file):
    # load preprocessors and the class
    with open(model_file, "rb") as f:
      preprocessors = pickle.load(f)
      myclass = pickle.load(f)
      _kwargs = pickle.load(f)

    myinstance = myclass(preprocessors=preprocessors)
    # just for consistent string representation
    myinstance._kwargs.update(_kwargs)

    d5 = bob.io.base.HDF5File(self._get_hdf5_file(model_file))
    try:
      myinstance.machine.load(d5)
    finally:
      d5.close()

    return myinstance
