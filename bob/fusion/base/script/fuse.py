"""A script to help for score fusion experiments
"""
from __future__ import absolute_import, division, print_function

import logging
import os
import sys

import click
import numpy as np

from bob.bio.base import utils
from bob.bio.base.score import dump_score, load_score
from bob.extension.scripts.click_helper import ResourceOption, verbosity_option

from ..tools import (
    check_consistency,
    get_2negatives_1positive,
    get_gza_from_lines_list,
    get_score_lines,
    get_scores,
    remove_nan,
)

logger = logging.getLogger(__name__)


def write_info(
    scores,
    algorithm,
    groups,
    output_dir,
    model_file,
    skip_check,
    force,
    **kwargs
):
    info = """
scores: %s
algorithm: %s
groups: %s
output_dir: %s
model_file: %s
skip_check: %s
force: %s
kwargs: %s
    """ % (
        scores,
        algorithm,
        groups,
        output_dir,
        model_file,
        skip_check,
        force,
        kwargs,
    )
    logger.debug(info)

    info_file = os.path.join(output_dir, "Experiment.info")
    with open(info_file, "w") as f:
        f.write("Command line:\n")
        f.write(str(sys.argv[1:]) + "\n\n")
        f.write("Configuration:\n\n")
        f.write(info)


def save_fused_scores(save_path, fused_scores, score_lines):
    score_lines["score"] = fused_scores
    gen, zei, atk, _, _, _ = get_2negatives_1positive(score_lines)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump_score(save_path, score_lines)
    dump_score(save_path + "-licit", np.append(gen, zei))
    dump_score(save_path + "-spoof", np.append(gen, atk))
    dump_score(save_path + "-real", np.append(gen, zei))
    dump_score(save_path + "-attack", atk)


def routine_fusion(
    algorithm,
    model_file,
    scores_train_lines,
    scores_train,
    train_neg,
    train_pos,
    fused_train_file,
    scores_dev_lines=None,
    scores_dev=None,
    dev_neg=None,
    dev_pos=None,
    fused_dev_file=None,
    scores_eval_lines=None,
    scores_eval=None,
    fused_eval_file=None,
    force=False,
    min_file_size=1000,
    do_training=True,
):

    # load the model if model_file exists and no training data was provided
    if os.path.exists(model_file) and not do_training:
        logger.info("Loading the algorithm from %s", model_file)
        algorithm = algorithm.load(model_file)

    # train the preprocessors
    if train_neg is not None and do_training:
        train_scores = np.vstack((train_neg, train_pos))
        neg_len = train_neg.shape[0]
        y = np.zeros((train_scores.shape[0],), dtype="bool")
        y[neg_len:] = True
        algorithm.train_preprocessors(train_scores, y)

    # preprocess data
    if scores_train is not None:
        scores_train = algorithm.preprocess(scores_train)
        train_neg, train_pos = algorithm.preprocess(
            train_neg
        ), algorithm.preprocess(train_pos)

    if scores_dev is not None:
        scores_dev = algorithm.preprocess(scores_dev)
        dev_neg, dev_pos = algorithm.preprocess(dev_neg), algorithm.preprocess(
            dev_pos
        )

    if scores_eval is not None:
        scores_eval = algorithm.preprocess(scores_eval)

    # Train the classifier
    if train_neg is not None and do_training:
        if utils.check_file(model_file, force, min_file_size):
            logger.info("model '%s' already exists.", model_file)
            algorithm = algorithm.load(model_file)
        else:
            algorithm.train(train_neg, train_pos, dev_neg, dev_pos)
            algorithm.save(model_file)

    # fuse the scores (train)
    if scores_train is not None:
        if utils.check_file(fused_train_file, force, min_file_size):
            logger.info("score file '%s' already exists.", fused_train_file)
        else:
            fused_scores_train = algorithm.fuse(scores_train)
            save_fused_scores(
                fused_train_file, fused_scores_train, scores_train_lines
            )

    # fuse the scores (dev)
    if scores_dev is not None:
        if utils.check_file(fused_dev_file, force, min_file_size):
            logger.info("score file '%s' already exists.", fused_dev_file)
        else:
            fused_scores_dev = algorithm.fuse(scores_dev)
            save_fused_scores(
                fused_dev_file, fused_scores_dev, scores_dev_lines
            )

    # fuse the scores (eval)
    if scores_eval is not None:
        if utils.check_file(fused_eval_file, force, min_file_size):
            logger.info("score file '%s' already exists.", fused_eval_file)
        else:
            fused_scores_eval = algorithm.fuse(scores_eval)
            save_fused_scores(
                fused_eval_file, fused_scores_eval, scores_eval_lines
            )


@click.command(
    epilog="""\b
Examples:
# normal score fusion using the mean algorithm:
$ bob fusion fuse -vvv sys1/scores-{world,dev,eval} sys2/scores-{world,dev,eval} -a mean
# same thing but more compact using bash expansion:
$ bob fusion fuse -vvv {sys1,sys2}/scores-{world,dev,eval} -a mean
# using an already trained algorithm:
$ bob fusion fuse -vvv {sys1,sys2}/scores-{dev,eval} -g dev -g eval -a mean -m /path/saved_model.pkl
# train an algorithm using development set scores:
$ bob fusion fuse -vvv {sys1,sys2}/scores-{dev,dev,eval} -a mean
# run fusion without eval scores:
$ bob fusion fuse -vvv {sys1,sys2}/scores-{world,dev} -g train -g dev -a mean
# run fusion with bio and pad systems:
$ bob fusion fuse -vvv sys_bio/scores-{world,dev,eval} sys_pad/scores-{train,dev,eval} -a mean
"""
)
@click.argument("scores", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--algorithm",
    "-a",
    required=True,
    cls=ResourceOption,
    entry_point_group="bob.fusion.algorithm",
    help="The fusion algorithm " "(:any:`bob.fusion.algorithm.Algorithm`).",
)
@click.option(
    "--groups",
    "-g",
    default=("train", "dev", "eval"),
    multiple=True,
    show_default=True,
    type=click.Choice(("train", "dev", "eval")),
    help="The groups of the scores. This should correspond to the "
    "scores that are provided. The order of options are important "
    "and should be in the same order as (train, dev, eval). Repeat "
    "this option for multiple values.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    default="fusion_result",
    show_default=True,
    type=click.Path(writable=True),
    help="The directory to save the annotations.",
)
@click.option(
    "--model-file",
    "-m",
    help="The path to where the algorithm will be saved/loaded.",
)
@click.option(
    "--skip-check",
    is_flag=True,
    show_default=True,
    help="If True, it will skip checking for "
    "the consistency between scores.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    show_default=True,
    help="Whether to overwrite existing files.",
)
@verbosity_option()
def fuse(
    scores,
    algorithm,
    groups,
    output_dir,
    model_file,
    skip_check,
    force,
    **kwargs
):
    """Score fusion

    The script takes several scores from different biometric and pad systems
    and does score fusion based on the scores and the algorithm provided.

    The scores are divided into 3 different sets: train, dev, and eval.
    Depending on which of these scores you provide, the script will skip parts
    of the execution. Provide train (and optionally dev) score files to train
    your algorithm.

    \b
    Raises
    ------
    click.BadArgumentUsage
        If the number of score files is not divisible by the number of groups.
    click.MissingParameter
        If the algorithm is not provided.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not model_file:
        do_training = True
        model_file = os.path.join(output_dir, "Model.pkl")
    else:
        do_training = False
    fused_train_file = os.path.join(output_dir, "scores-train")
    fused_dev_file = os.path.join(output_dir, "scores-dev")
    fused_eval_file = os.path.join(output_dir, "scores-eval")

    if not len(scores) % len(groups) == 0:
        raise click.BadArgumentUsage(
            "The number of scores must be a multiple of the number of groups."
        )

    if algorithm is None:
        raise click.MissingParameter(
            "algorithm must be provided.", param_type="option"
        )

    write_info(
        scores,
        algorithm,
        groups,
        output_dir,
        model_file,
        skip_check,
        force,
        **kwargs
    )

    """Do the actual fusion."""

    train_files, dev_files, eval_files = [], [], []
    for i, (files, grp) in enumerate(
        zip((train_files, dev_files, eval_files), ("train", "dev", "eval"))
    ):
        try:
            idx = groups.index(grp)
            files.extend(scores[idx :: len(groups)])
        except ValueError:
            pass

    click.echo("train_files: %s" % train_files)
    click.echo("dev_files: %s" % dev_files)
    click.echo("eval_files: %s" % eval_files)

    # load the scores
    if train_files:
        score_lines_list_train = [load_score(path) for path in train_files]
    if dev_files:
        score_lines_list_dev = [load_score(path) for path in dev_files]
    if eval_files:
        score_lines_list_eval = [load_score(path) for path in eval_files]

    # genuine, zero effort impostor, and attack list of
    # train, development and evaluation data.
    if train_files:
        _, gen_lt, zei_lt, atk_lt = get_gza_from_lines_list(
            score_lines_list_train
        )
    if dev_files:
        _, gen_ld, zei_ld, atk_ld = get_gza_from_lines_list(
            score_lines_list_dev
        )
    if eval_files:
        _, gen_le, zei_le, atk_le = get_gza_from_lines_list(
            score_lines_list_eval
        )

    # check if score lines are consistent
    if not skip_check:
        if train_files:
            logger.info("Checking the training files for consistency ...")
            check_consistency(gen_lt, zei_lt, atk_lt)
        if dev_files:
            logger.info("Checking the development files for consistency ...")
            check_consistency(gen_ld, zei_ld, atk_ld)
        if eval_files:
            logger.info("Checking the evaluation files for consistency ...")
            check_consistency(gen_le, zei_le, atk_le)

    if train_files:
        scores_train = get_scores(gen_lt, zei_lt, atk_lt)
        scores_train_lines = get_score_lines(
            gen_lt[0:1], zei_lt[0:1], atk_lt[0:1]
        )
        train_neg = get_scores(zei_lt, atk_lt)
        train_pos = get_scores(gen_lt)
    else:
        scores_train, scores_train_lines, train_neg, train_pos = (
            None,
            None,
            None,
            None,
        )

    if dev_files:
        scores_dev = get_scores(gen_ld, zei_ld, atk_ld)
        scores_dev_lines = get_score_lines(
            gen_ld[0:1], zei_ld[0:1], atk_ld[0:1]
        )
        dev_neg = get_scores(zei_ld, atk_ld)
        dev_pos = get_scores(gen_ld)
    else:
        scores_dev, scores_dev_lines, dev_neg, dev_pos = None, None, None, None

    if eval_files:
        scores_eval = get_scores(gen_le, zei_le, atk_le)
        scores_eval_lines = get_score_lines(
            gen_le[0:1], zei_le[0:1], atk_le[0:1]
        )
    else:
        scores_eval, scores_eval_lines = None, None

    # check for nan values
    found_nan = False
    if train_files:
        found_nan, nan_train, scores_train = remove_nan(scores_train, found_nan)
        scores_train_lines = scores_train_lines[~nan_train]
        found_nan, _, train_neg = remove_nan(train_neg, found_nan)
        found_nan, _, train_pos = remove_nan(train_pos, found_nan)
    if dev_files:
        found_nan, nan_dev, scores_dev = remove_nan(scores_dev, found_nan)
        scores_dev_lines = scores_dev_lines[~nan_dev]
        found_nan, _, dev_neg = remove_nan(dev_neg, found_nan)
        found_nan, _, dev_pos = remove_nan(dev_pos, found_nan)
    if eval_files:
        found_nan, nan_eval, scores_eval = remove_nan(scores_eval, found_nan)
        scores_eval_lines = scores_eval_lines[~nan_eval]

    if found_nan:
        logger.warning("Some nan values were removed.")

    routine_fusion(
        algorithm,
        model_file,
        scores_train_lines,
        scores_train,
        train_neg,
        train_pos,
        fused_train_file,
        scores_dev_lines,
        scores_dev,
        dev_neg,
        dev_pos,
        fused_dev_file,
        scores_eval_lines,
        scores_eval,
        fused_eval_file,
        force,
        do_training=do_training,
    )
