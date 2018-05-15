#!/usr/bin/env python

import numpy
import bob.learn.em


def grouping(scores, gformat='random', npoints=500, seed=None, **kwargs):

    scores = numpy.asarray(scores)
    if scores.size == 0:
        return scores

    if(gformat == "kmeans"):
        kmeans_machine = bob.learn.em.KMeansMachine(npoints, 2)
        kmeans_trainer = bob.learn.em.KMeansTrainer()
        bob.learn.em.train(
            kmeans_trainer, kmeans_machine, scores, max_iterations=500,
            convergence_threshold=0.1)
        scores = kmeans_machine.means

    elif(gformat == "random"):
        if seed is not None:
            numpy.random.seed(seed)
        scores_indexes = numpy.array(
            numpy.random.rand(npoints) * scores.shape[0], dtype=int)

        scores = scores[scores_indexes]

    return scores
