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

logger = bob.core.log.setup("bob.fusion.base")


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
  algorithm.plot_boundary_decision(
    scores, score_labels, threshold,
    do_grouping=True,
    npoints=int(args['--group']),
    seed=0,
    gformat=args['--grouping']
  )
  plt.savefig(args['--output'])
  plt.close()

if __name__ == '__main__':
  main()
