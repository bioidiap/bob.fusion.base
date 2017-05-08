#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Amir Mohammadi <amir.mohammadi@idiap.ch>

from __future__ import print_function, absolute_import, division

import os
import numpy as np

from bob.io.base import create_directories_safe
from bob.measure.load import load_score, dump_score
from bob.bio.base import utils

from ..tools import parse_arguments, write_info, get_gza_from_lines_list, \
    check_consistency, get_scores, remove_nan, get_score_lines, \
    get_2negatives_1positive

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


def save_fused_scores(save_path, fused_scores, score_lines):
    score_lines['score'] = fused_scores
    gen, zei, atk, _, _, _ = get_2negatives_1positive(score_lines)
    create_directories_safe(os.path.dirname(save_path))
    dump_score(save_path, score_lines)
    dump_score(save_path + '-licit', np.append(gen, zei))
    dump_score(save_path + '-spoof', np.append(gen, atk))
    dump_score(save_path + '-real', np.append(gen, zei))
    dump_score(save_path + '-attack', atk)


def routine_fusion(
        algorithm, model_file,
        scores_train_lines, scores_train, train_neg, train_pos,
        fused_train_file,
        scores_dev_lines=None, scores_dev=None, dev_neg=None, dev_pos=None,
        fused_dev_file=None,
        scores_eval_lines=None, scores_eval=None, fused_eval_file=None,
        force=False, min_file_size=1000):
    # train the preprocessors
    train_scores = np.vstack((train_neg, train_pos))
    neg_len = train_neg.shape[0]
    y = np.zeros((train_scores.shape[0],), dtype='bool')
    y[neg_len:] = True
    algorithm.train_preprocessors(train_scores, y)

    # preprocess data
    scores_train = algorithm.preprocess(scores_train)
    train_neg, train_pos = algorithm.preprocess(train_neg), algorithm.preprocess(train_pos)
    if scores_dev is not None:
        scores_dev = algorithm.preprocess(scores_dev)
        dev_neg, dev_pos = algorithm.preprocess(dev_neg), algorithm.preprocess(dev_pos)
    if scores_eval is not None:
        scores_eval = algorithm.preprocess(scores_eval)

    # train the model
    if utils.check_file(model_file, force, min_file_size):
        logger.info(
            "model '%s' already exists.", model_file)
        algorithm = algorithm.load(model_file)
    else:
        algorithm.train(train_neg, train_pos, dev_neg, dev_pos)
        algorithm.save(model_file)

    # fuse the scores (train)
    if utils.check_file(fused_train_file, force, min_file_size):
        logger.info(
            "score file '%s' already exists.", fused_train_file)
    else:
        fused_scores_train = algorithm.fuse(scores_train)
        save_fused_scores(fused_train_file, fused_scores_train, scores_train_lines)

    if scores_dev is not None:
        # fuse the scores (dev)
        if utils.check_file(fused_dev_file, force, min_file_size):
            logger.info(
                "score file '%s' already exists.", fused_dev_file)
        else:
            fused_scores_dev = algorithm.fuse(scores_dev)
            save_fused_scores(fused_dev_file, fused_scores_dev, scores_dev_lines)

    if scores_eval is not None:
        # fuse the scores (eval)
        if utils.check_file(fused_eval_file, force, min_file_size):
            logger.info(
                "score file '%s' already exists.", fused_eval_file)
        else:
            fused_scores_eval = algorithm.fuse(scores_eval)
            save_fused_scores(fused_eval_file, fused_scores_eval, scores_eval_lines)


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
        logger.info('Checking the training files for consistency ...')
        check_consistency(gen_lt, zei_lt, atk_lt)
        if args.dev_files:
            logger.info('Checking the development files for consistency ...')
            check_consistency(gen_ld, zei_ld, atk_ld)
        if args.eval_files:
            logger.info('Checking the evaluation files for consistency ...')
            check_consistency(gen_le, zei_le, atk_le)

    scores_train = get_scores(gen_lt, zei_lt, atk_lt)
    scores_train_lines = get_score_lines(gen_lt[0:1], zei_lt[0:1], atk_lt[0:1])
    train_neg = get_scores(zei_lt, atk_lt)
    train_pos = get_scores(gen_lt)
    if args.dev_files:
        scores_dev = get_scores(gen_ld, zei_ld, atk_ld)
        scores_dev_lines = get_score_lines(gen_ld[0:1], zei_ld[0:1], atk_ld[0:1])
        dev_neg = get_scores(zei_ld, atk_ld)
        dev_pos = get_scores(gen_ld)
    else:
        scores_dev, scores_dev_lines, dev_neg, dev_pos = None, None, None, None
    if args.eval_files:
        scores_eval = get_scores(gen_le, zei_le, atk_le)
        scores_eval_lines = get_score_lines(gen_le[0:1], zei_le[0:1], atk_le[0:1])
    else:
        scores_eval, scores_eval_lines = None, None

    # check for nan values
    found_nan = False
    found_nan, nan_train, scores_train = remove_nan(scores_train, found_nan)
    scores_train_lines = scores_train_lines[~nan_train]
    found_nan, _, train_neg = remove_nan(train_neg, found_nan)
    found_nan, _, train_pos = remove_nan(train_pos, found_nan)
    if args.dev_files:
        found_nan, nan_dev, scores_dev = remove_nan(scores_dev, found_nan)
        scores_dev_lines = scores_dev_lines[~nan_dev]
        found_nan, _, dev_neg = remove_nan(dev_neg, found_nan)
        found_nan, _, dev_pos = remove_nan(dev_pos, found_nan)
    if args.eval_files:
        found_nan, nan_eval, scores_eval = remove_nan(scores_eval, found_nan)
        scores_eval_lines = scores_eval_lines[~nan_eval]

    if found_nan:
        logger.warn('Some nan values were removed.')

    routine_fusion(
        algorithm, args.model_file, scores_train_lines, scores_train,
        train_neg, train_pos, args.fused_train_file, scores_dev_lines,
        scores_dev, dev_neg, dev_pos, args.fused_dev_file, scores_eval_lines,
        scores_eval, args.fused_eval_file, args.force)


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
