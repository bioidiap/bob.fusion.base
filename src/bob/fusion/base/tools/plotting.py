#!/usr/bin/env python

import numpy as np

from numpy.random import default_rng

from bob.learn.em import KMeansMachine


def grouping(scores, gformat="random", npoints=500, seed=None, **kwargs):
    scores = np.asarray(scores)
    if scores.size == 0:
        return scores

    if gformat == "kmeans":
        kmeans_machine = KMeansMachine(
            n_clusters=npoints, convergence_threshold=0.1, max_iter=500
        )
        kmeans_machine.fit(scores)
        scores = kmeans_machine.means

    elif gformat == "random":
        rng = default_rng(seed)
        scores = rng.choice(scores, npoints, replace=False)

    return scores
