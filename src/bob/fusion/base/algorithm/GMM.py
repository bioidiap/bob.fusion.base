#!/usr/bin/env python

from __future__ import absolute_import, division

import logging

import numpy as np

from bob.learn.em import GMMMachine

from .AlgorithmBob import AlgorithmBob

logger = logging.getLogger("bob.fusion.base")


class GMM(AlgorithmBob):
    """GMM Score fusion"""

    def __init__(
        self,
        # parameters for the GMM
        number_of_gaussians=None,
        # parameters of GMM training
        # Maximum number of iterations for ML GMM Training
        gmm_training_iterations=25,
        # Threshold to end the ML training
        training_threshold=5e-4,
        # Minimum value that a variance can reach
        variance_threshold=5e-4,
        update_weights=True,
        update_means=True,
        update_variances=True,
        init_seed=5489,
        **kwargs,
    ):
        super().__init__(classifier=self, **kwargs)
        self.str["number_of_gaussians"] = number_of_gaussians
        self.str["gmm_training_iterations"] = gmm_training_iterations
        self.str["training_threshold"] = training_threshold
        self.str["variance_threshold"] = variance_threshold
        self.str["update_weights"] = update_weights
        self.str["update_means"] = update_means
        self.str["update_variances"] = update_variances
        self.str["init_seed"] = init_seed

        # copy parameters
        self.n_gaussians = number_of_gaussians
        self.gmm_training_iterations = gmm_training_iterations
        self.training_threshold = training_threshold
        self.variance_threshold = variance_threshold
        self.update_weights = update_weights
        self.update_means = update_means
        self.update_variances = update_variances
        self.init_seed = init_seed

        # this is needed to be able to load the machine
        self.machine = GMMMachine(n_gaussians=1)

    def train(self, train_neg, train_pos, devel_neg=None, devel_pos=None):
        logger.info("Using only positive samples for training")
        array = train_pos
        logger.debug("Training files have the shape of {}".format(array.shape))

        if self.n_gaussians is None:
            self.n_gaussians = array.shape[1] + 1
            logger.warning(
                "Number of Gaussians was None. "
                "Using {}.".format(self.n_gaussians)
            )

        # Creates the machines (KMeans and GMM)
        logger.debug("Training GMM machine")
        self.machine = GMMMachine(
            n_gaussians=self.n_gaussians,
            convergence_threshold=self.training_threshold,
            max_fitting_steps=self.gmm_training_iterations,
            random_state=self.init_seed,
            update_means=self.update_means,
            update_variances=self.update_variances,
            update_weights=self.update_weights,
        )
        self.machine.fit(array)

    def decision_function(self, scores):
        return np.fromiter(
            (self.machine.log_likelihood(s) for s in scores),
            np.float,
            scores.shape[0],
        )
