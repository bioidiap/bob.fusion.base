#!/usr/bin/env python

from __future__ import absolute_import, division

import logging
import pickle

import numpy as np

logger = logging.getLogger(__name__)


class Algorithm(object):
    """A class to be used in score fusion

    Attributes
    ----------
    classifier
    preprocessors
    str : dict
        A dictionary that its content will printed in the __str__ method.
    """

    def __init__(self, preprocessors=None, classifier=None, **kwargs):
        """
        Parameters
        ----------
        preprocessors : :any:`list`
            An optional list of preprocessors that follow the API of
            :any:`sklearn.preprocessing.StandardScaler`. Especially
            `fit_transform` and `transform` must be implemented.
        classifier
            An instance of a class that implements `fit(X[, y])` and
            `decision_function(X)` like:
            :any:`sklearn.linear_model.LogisticRegression`
        **kwargs
            All extra
        """
        super(Algorithm, self).__init__(**kwargs)
        self.classifier = classifier
        self.preprocessors = preprocessors
        self.str = {"preprocessors": preprocessors}
        if classifier is not self:
            self.str["classifier"] = classifier

    def train_preprocessors(self, X, y=None):
        """Train preprocessors in order.
        X: numpy.ndarray with the shape of (n_samples, n_systems)."""
        if self.preprocessors is not None:
            for preprocessor in self.preprocessors:
                X = preprocessor.fit_transform(X, y)

    def preprocess(self, scores):
        """
        scores: numpy.ndarray with the shape of (n_samples, n_systems).
        returns the transformed scores."""
        if scores.size == 0:
            return scores
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
        y = np.zeros((train_scores.shape[0],), dtype="bool")
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
        return "%s(%s)" % (
            str(self.__class__),
            ", ".join(
                [
                    "%s=%s" % (key, value)
                    for key, value in self.str.items()
                    if value is not None
                ]
            ),
        )

    def save(self, model_file):
        """Save the instance of the algorithm.

        model_file: str
            A path to save the file. Please note that file objects
            are not accepted. The filename MUST end with ".pkl".
            Also, an algorithm may save itself in multiple files with different
            extensions such as model_file and model_file[:-3]+'hdf5'.
        """
        # support for bob machines
        if hasattr(self, "custom_save"):
            self.custom_save(model_file)
        else:
            with open(model_file, "wb") as f:
                pickle.dump(type(self), f)
                pickle.dump(self, f)

    def load(self, model_file):
        """Load the algorithm the same way it was saved.
        A new instance will be returned.

        **Returns:**

        loaded_algorithm: Algorithm
            A new instance of the loaded algorithm.
        """
        with open(model_file, "rb") as f:
            algo_class = pickle.load(f)
            algo = algo_class()
            if not hasattr(algo, "custom_save"):
                return pickle.load(f)
        return algo.load(model_file)
