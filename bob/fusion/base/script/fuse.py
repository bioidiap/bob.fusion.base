#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Amir Mohammadi <amir.mohammadi@idiap.ch>

from __future__ import print_function, absolute_import, division

import os
import numpy as np

from bob.io.base import create_directories_safe
from bob.measure.load import load_score, get_all_scores,\
    get_negatives_positives_all, dump_score
from bob.bio.base import utils

from ..tools import parse_arguments, write_info

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


def fuse(args, command_line_parameters):
  """Do the actual fusion."""
  algorithm = args.algorithm

  write_info(args, command_line_parameters)

  # load the scores
  score_lines_list_dev = [load_score(path, ncolumns=args.score_type)
                          for path in args.dev_files]
  scores_dev = get_all_scores(score_lines_list_dev)
  trainer_scores = get_negatives_positives_all(score_lines_list_dev)
  if args.eval_files:
    score_lines_list_eval = [load_score(path, ncolumns=args.score_type)
                             for path in args.eval_files]
    scores_eval = get_all_scores(score_lines_list_eval)
  else:
    score_lines_list_eval = []
    scores_eval = []

  # check if score lines are consistent
  if not args.skip_check:
    score_lines0 = score_lines_list_dev[0]
    for score_lines in score_lines_list_dev[1:]:
      assert(np.all(score_lines['claimed_id'] == score_lines0['claimed_id']))
      assert(np.all(score_lines['real_id'] == score_lines0['real_id']))
    if args.eval_files:
      score_lines0 = score_lines_list_eval[0]
      for score_lines in score_lines_list_eval[1:]:
        assert(np.all(score_lines['claimed_id'] == score_lines0['claimed_id']))
        assert(np.all(score_lines['real_id'] == score_lines0['real_id']))

  # train the preprocessors
  algorithm.train_preprocessors(scores_dev)

  # preprocess data
  scores_dev = algorithm.preprocess(scores_dev)
  scores_eval = algorithm.preprocess(scores_eval)
  neg, pos = trainer_scores
  neg, pos = algorithm.preprocess(neg), algorithm.preprocess(pos)
  trainer_scores = (neg, pos)

  # train the model
  if utils.check_file(args.model_file, args.force, 1000):
    logger.info(
      "- Fusion: model '%s' already exists.", args.model_file)
    algorithm = algorithm.load(args.model_file)
  else:
    algorithm.train(trainer_scores)
    algorithm.save(args.model_file)

  # fuse the scores (dev)
  if utils.check_file(args.fused_dev_file, args.force, 1000):
    logger.info(
      "- Fusion: scores '%s' already exists.", args.fused_dev_file)
  else:
    fused_scores_dev = algorithm.fuse(scores_dev)
    score_lines = score_lines_list_dev[0]
    score_lines['score'] = fused_scores_dev
    create_directories_safe(os.path.dirname(args.fused_dev_file))
    dump_score(args.fused_dev_file, score_lines)

  # fuse the scores (eval)
  if args.eval_files:
    if utils.check_file(args.fused_eval_file, args.force, 1000):
      logger.info(
        "- Fusion: scores '%s' already exists.", args.fused_eval_file)
    else:
      fused_scores_eval = algorithm.fuse(scores_eval)
      score_lines = score_lines_list_eval[0]
      score_lines['score'] = fused_scores_eval
      create_directories_safe(os.path.dirname(args.fused_eval_file))
      dump_score(args.fused_eval_file, score_lines)


def main(command_line_parameters=None):
  """Executes the main function"""
  try:
    # do the command line parsing
    args = parse_arguments(command_line_parameters)

    # perform face verification test
    fuse(args, command_line_parameters)
  except Exception as e:
    # track any exceptions as error logs (i.e., to get a time stamp)
    logger.error("During the execution, an exception was raised: %s" % e)
    raise

if __name__ == "__main__":
  main()
