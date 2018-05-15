#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import bob.learn.em
import numpy

from .AlgorithmBob import AlgorithmBob

import logging
logger = logging.getLogger("bob.fusion.base")


class GMM(AlgorithmBob):
    """GMM Score fusion"""

    def __init__(self,
                 # parameters for the GMM
                 number_of_gaussians=None,
                 # parameters of GMM training
                 # Maximum number of iterations for K-Means
                 kmeans_training_iterations=25,
                 # Maximum number of iterations for ML GMM Training
                 gmm_training_iterations=25,
                 # Threshold to end the ML training
                 training_threshold=5e-4,
                 # Minimum value that a variance can reach
                 variance_threshold=5e-4,
                 update_weights=True,
                 update_means=True,
                 update_variances=True,
                 # If set, the weight of a particular Gaussian will at least be
                 # greater than this threshold. In the case the real weight is
                 # lower, the prior mean value will be used to estimate the
                 # current mean and variance.
                 responsibility_threshold=0,
                 init_seed=5489,
                 **kwargs):
        super(GMM, self).__init__(classifier=self, **kwargs)
        self.str['number_of_gaussians'] = number_of_gaussians
        self.str['kmeans_training_iterations'] = kmeans_training_iterations
        self.str['gmm_training_iterations'] = gmm_training_iterations
        self.str['training_threshold'] = training_threshold
        self.str['variance_threshold'] = variance_threshold
        self.str['update_weights'] = update_weights
        self.str['update_means'] = update_means
        self.str['update_variances'] = update_variances
        self.str['responsibility_threshold'] = responsibility_threshold
        self.str['init_seed'] = init_seed

        # copy parameters
        self.gaussians = number_of_gaussians
        self.kmeans_training_iterations = kmeans_training_iterations
        self.gmm_training_iterations = gmm_training_iterations
        self.training_threshold = training_threshold
        self.variance_threshold = variance_threshold
        self.update_weights = update_weights
        self.update_means = update_means
        self.update_variances = update_variances
        self.responsibility_threshold = responsibility_threshold
        self.init_seed = init_seed
        self.rng = bob.core.random.mt19937(self.init_seed)

        # this is needed to be able to load the machine
        self.machine = bob.learn.em.GMMMachine()
        self.kmeans_trainer = bob.learn.em.KMeansTrainer()
        self.gmm_trainer = bob.learn.em.ML_GMMTrainer(
            self.update_means, self.update_variances, self.update_weights,
            self.responsibility_threshold)

    def train(self, train_neg, train_pos, devel_neg=None, devel_pos=None):
        logger.info("Using only positive samples for training")
        array = train_pos
        logger.debug("Training files have the shape of {}".format(array.shape))

        if self.gaussians is None:
            self.gaussians = array.shape[1] + 1
            logger.warn("Number of Gaussians was None. "
                        "Using {}.".format(self.gaussians))

        # Computes input size
        input_size = array.shape[1]

        # Creates the machines (KMeans and GMM)
        logger.debug("Creating machines")
        kmeans = bob.learn.em.KMeansMachine(self.gaussians, input_size)
        self.machine = bob.learn.em.GMMMachine(self.gaussians, input_size)

        # Trains using the KMeansTrainer
        logger.info("Training K-Means")
        bob.learn.em.train(self.kmeans_trainer, kmeans, array,
                           self.kmeans_training_iterations,
                           self.training_threshold, self.rng)

        variances, weights = \
            kmeans.get_variances_and_weights_for_each_cluster(array)
        means = kmeans.means

        # Initializes the GMM
        self.machine.means = means
        self.machine.variances = variances
        self.machine.weights = weights
        self.machine.set_variance_thresholds(self.variance_threshold)

        # Trains the GMM
        logger.info("Training GMM")
        bob.learn.em.train(self.gmm_trainer, self.machine, array,
                           self.gmm_training_iterations,
                           self.training_threshold, self.rng)

    def decision_function(self, scores):
        return numpy.fromiter((self.machine(s) for s in scores),
                              numpy.float, scores.shape[0])
