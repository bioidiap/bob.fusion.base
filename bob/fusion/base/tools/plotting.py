#!/usr/bin/env python

import bob.learn.em
import numpy as np
from numpy.random import default_rng


def grouping(scores, gformat="random", npoints=500, seed=None, **kwargs):

    scores = np.asarray(scores)
    if scores.size == 0:
        return scores

    if gformat == "kmeans":
        kmeans_machine = bob.learn.em.KMeansMachine(npoints, 2)
        kmeans_trainer = bob.learn.em.KMeansTrainer()
        bob.learn.em.train(
            kmeans_trainer,
            kmeans_machine,
            scores,
            max_iterations=500,
            convergence_threshold=0.1,
        )
        scores = kmeans_machine.means

    elif gformat == "random":
        rng = default_rng(seed)
        scores = rng.choice(scores, npoints, replace=False)

    return scores
