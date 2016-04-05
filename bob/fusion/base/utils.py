#!/usr/bin/env python

import bob.learn.em
import numpy


def grouping(negatives, positives,
             gformat='random', npoints=500, seed=None, **kwargs):

  negatives = numpy.asarray(negatives)
  positives = numpy.asarray(positives)

  if(gformat == "kmeans"):

    kmeans_negatives = bob.learn.em.KMeansMachine(npoints, 2)
    kmeans_positives = bob.learn.em.KMeansMachine(npoints, 2)

    kmeansTrainer = bob.learn.em.KMeansTrainer()

    bob.learn.em.train(
      kmeansTrainer, kmeans_negatives, negatives, max_iterations=500,
      convergence_threshold=0.1)
    bob.learn.em.train(
      kmeansTrainer, kmeans_positives, positives, max_iterations=500,
      convergence_threshold=0.1)

    negatives = kmeans_negatives.means
    positives = kmeans_positives.means

  elif(gformat == "random"):
    if seed is not None:
      numpy.random.seed(seed)
    negatives_indexes = numpy.array(
      numpy.random.rand(npoints) * negatives.shape[0], dtype=int)
    positives_indexes = numpy.array(
      numpy.random.rand(npoints) * positives.shape[0], dtype=int)

    negatives = negatives[negatives_indexes]
    positives = positives[positives_indexes]

  return negatives, positives
