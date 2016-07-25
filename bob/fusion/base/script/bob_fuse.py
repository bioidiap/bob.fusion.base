#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Amir Mohammadi <amir.mohammadi@idiap.ch>

from __future__ import print_function, absolute_import, division

import os

from bob.io.base import create_directories_safe
from bob.measure.load import load_score, dump_score
from bob.bio.base import utils

from ..tools import parse_arguments, write_info, get_gza_from_lines_list, \
   check_consistency, get_scores, remove_nan

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


def fuse(args, command_line_parameters):
  """Do the actual fusion."""
  algorithm = args.algorithm

  write_info(args, command_line_parameters)

  # load the scores
  score_lines_list_train = [load_score(path, ncolumns=args.score_type)
                            for path in args.train_files]
  if args.dev_files:
    score_lines_list_dev = [load_score(path, ncolumns=args.score_type)
                            for path in args.dev_files]
  if args.eval_files:
    score_lines_list_eval = [load_score(path, ncolumns=args.score_type)
                             for path in args.eval_files]

  # genuine, zero effort impostor, and attack list of
  # train, development and evaluation data.
  idx1, gen_lt, zei_lt, atk_lt = get_gza_from_lines_list(score_lines_list_train)
  if args.dev_files:
    _, gen_ld, zei_ld, atk_ld = get_gza_from_lines_list(score_lines_list_dev)
  if args.eval_files:
    _, gen_le, zei_le, atk_le = get_gza_from_lines_list(score_lines_list_eval)

  # check if score lines are consistent
  if not args.skip_check:
    check_consistency(gen_lt, zei_lt, atk_lt)
    if args.dev_files:
      check_consistency(gen_ld, zei_ld, atk_ld)
    if args.eval_files:
      check_consistency(gen_le, zei_le, atk_le)

  scores_train = get_scores(gen_lt, zei_lt, atk_lt)
  train_neg = get_scores(zei_lt, atk_lt)
  train_pos = get_scores(gen_lt)
  if args.dev_files:
    scores_dev = get_scores(gen_ld, zei_ld, atk_ld)
    dev_neg = get_scores(zei_ld, atk_ld)
    dev_pos = get_scores(gen_ld)
  else:
    dev_neg, dev_pos = None, None
  if args.eval_files:
    scores_eval = get_scores(gen_le, zei_le, atk_le)

  # check for nan values
  found_nan = False
  found_nan, _, scores_train = remove_nan(scores_train, found_nan)
  found_nan, _, train_neg = remove_nan(train_neg, found_nan)
  found_nan, _, train_pos = remove_nan(train_pos, found_nan)
  if args.dev_files:
    found_nan, nan_dev, scores_dev = remove_nan(scores_dev, found_nan)
    found_nan, _, dev_neg = remove_nan(dev_neg, found_nan)
    found_nan, _, dev_pos = remove_nan(dev_pos, found_nan)
  if args.eval_files:
    found_nan, nan_eval, scores_eval = remove_nan(scores_eval, found_nan)

  if found_nan:
    logger.warn('Some nan values were removed.')

  # train the preprocessors
  algorithm.train_preprocessors(scores_train)

  # preprocess data
  train_neg, train_pos = algorithm.preprocess(train_neg), algorithm.preprocess(train_pos)
  if args.dev_files:
    scores_dev = algorithm.preprocess(scores_dev)
    dev_neg, dev_pos = algorithm.preprocess(dev_neg), algorithm.preprocess(dev_pos)
  if args.eval_files:
    scores_eval = algorithm.preprocess(scores_eval)

  # train the model
  if utils.check_file(args.model_file, args.force, 1000):
    logger.info(
      "- Fusion: model '%s' already exists.", args.model_file)
    algorithm = algorithm.load(args.model_file)
  else:
    algorithm.train(train_neg, train_pos, dev_neg, dev_pos)
    algorithm.save(args.model_file)

  # fuse the scores (dev)
  if utils.check_file(args.fused_dev_file, args.force, 1000):
    logger.info(
      "- Fusion: scores '%s' already exists.", args.fused_dev_file)
  elif args.dev_files:
    fused_scores_dev = algorithm.fuse(scores_dev)
    score_lines = score_lines_list_dev[idx1][~nan_dev]
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
      score_lines = score_lines_list_eval[idx1][~nan_eval]
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
