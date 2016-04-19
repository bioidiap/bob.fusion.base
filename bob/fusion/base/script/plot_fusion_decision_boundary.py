#!/usr/bin/env python

"""Plot decision boundraries of the fusion algorithm.

Usage:
  plot_fusion_decision_boundary.py SCORE_FILE SCORE_FILE MODEL_FILE
    [-v... | --verbose...] [options]
  plot_fusion_decision_boundary.py (-h | --help)
  plot_fusion_decision_boundary.py (-V | --version)

Options:
  -o, --output PLOT_FILE  The path to save the plot. [default: scatter.pdf]
  --score-type {4,5}      The format the scores are provided. [default: 4]
  -v, --verbose           Increase the verbosity level from 0 (only error
                          messages) to 1 (warnings), 2 (log messages), 3 (debug
                          information) by adding the --verbose option as often
                          as desired (e.g. '-vvv' for debug). [default: 0]
  -a, --algorithm Algorithm  The fusion that was used during fusion if they
                          implement a different load method e.g.
                          bob.fusion.base.algorithm.MLP.
                          [default: bob.fusion.base.algorithm.Algorithm]
  -g, --group N           If given scores will be grouped into N samples.
                          [default: 500]
  --grouping {random, kmeans}  The gouping algorithm used. [default: kmeans]
  -h --help               Show this screen.
  -V, --version           Show version.

"""

from docopt import docopt
import matplotlib.pyplot as plt
import numpy

import bob.fusion.base
import bob.core
from bob.measure.load import load_score, get_negatives_positives,\
  get_all_scores
from bob.measure import eer_threshold
from ..tools import grouping

logger = bob.core.log.setup("bob.fusion.base")


def plot_boundary_decision(algorithm, scores, score_labels, threshold,
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
  xx, yy = numpy.meshgrid(
    numpy.linspace(x_min, x_max, resolution),
    numpy.linspace(y_min, y_max, resolution))
  temp = numpy.c_[xx.ravel(), yy.ravel()]
  temp = algorithm.preprocess(temp)
  Z = (algorithm.fuse(temp) > threshold).reshape(xx.shape)

  contourf = plt.contour(xx, yy, Z, 1, alpha=1, cmap=plt.cm.viridis)

  if do_grouping:
    negatives, positives = X[numpy.logical_not(Y)], X[Y]
    negatives, positives = grouping(negatives, positives, **kwargs)
    X = numpy.concatenate((negatives, positives), axis=0)
    Y = numpy.concatenate(
      (numpy.zeros(negatives.shape[0], dtype=numpy.bool8),
       numpy.ones(positives.shape[0], dtype=numpy.bool8)),
      axis=0)

  negatives, positives = X[numpy.logical_not(Y)], X[Y]
  colors = plt.cm.viridis(numpy.linspace(0, 1, 2))
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


def main(command_line_parameters=None):
  args = docopt(__doc__, argv=command_line_parameters,
                version=bob.fusion.base.get_config())
  print(args)
  bob.core.log.set_verbosity_level(logger, args['--verbose'])

  # load the algorithm
  algorithm = eval('{}()'.format(args['--algorithm']))
  algorithm = algorithm.load(args['MODEL_FILE'])

  # load the scores
  score_lines_list = [
    load_score(path, int(args['--score-type'])) for path in args['SCORE_FILE']]
  scores = get_all_scores(score_lines_list)
  score_lines = numpy.array(score_lines_list[0])
  score_lines['score'] = algorithm.fuse(algorithm.preprocess(scores))
  threshold = eer_threshold(*get_negatives_positives(score_lines))
  score_labels = score_lines['claimed_id'] == score_lines['real_id']

  # plot the decision boundary
  plot_boundary_decision(
    algorithm, scores, score_labels, threshold,
    do_grouping=True,
    npoints=int(args['--group']),
    seed=0,
    gformat=args['--grouping']
  )
  plt.savefig(args['--output'])
  plt.close()

if __name__ == '__main__':
  main()
