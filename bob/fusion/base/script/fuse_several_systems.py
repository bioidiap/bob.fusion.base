#!/usr/bin/env python

"""Plot decision boundraries of the fusion algorithm."""

import argparse

import bob.fusion.base
import bob.core
import os
import sys

import bob.fusion.base.script.bob_fuse

logger = bob.core.log.setup("bob.fusion.base")


def main(command_line_parameters=None):
    basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
    OUTPUT_DIR = os.path.join(basedir, 'fusion_results')
    IN_DIR = "/Users/pavelkor/Documents/pav/idiap/src/results/"

    # setup command line parameters
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--prefix-path', default=IN_DIR,
                        help="The common path to where all scores are located (defaults to '%(default)s')")

    parser.add_argument('-s', '--system-names', nargs='+', required=True,
                        help='A list of system names to fuse.')

    parser.add_argument('-t', '--system-types', nargs='+', required=True,
                        help='Types of systems corresponding to names.', choices=('pad', 'asv'))

    parser.add_argument('-o', '--out-directory', default=OUTPUT_DIR,
                        help="This path will be prepended to every file output by this procedure (defaults to '%(default)s')")

    parser.add_argument('-f', '--fusion-type', default='parallel', required=True,
                        help="Which type of fusion to perform (defaults to '%(default)s')", choices=('parallel', 'cascade'))

    parser.add_argument('-a', '--fusion-algorithm', default='and', required=True,
                        help="The type of algorithm used for fusion (defaults to '%(default)s')",
                        choices=('and', 'llr', 'mlp', 'plr-2', 'mean'))

    parser.add_argument('-d', '--dataset-name-train', required=True,
                        help='The name of the dataset which was used to compute TRAIN and DEV scores.')

    parser.add_argument('-D', '--dataset-name-eval', required=True,
                        help='The name of the dataset which was used to compute EVAL scores.')

    parser.add_argument('-A', '--dataset-name-asv', required=False,
                        help='The name of the dataset that was used for ASV, specifically.')

    parser.add_argument('-g', '--score-group', nargs='+', default='dev', required=True,
                        help="Which set(s) of scores we fuse (defaults to '%(default)s')",
                        choices=('dev', 'eval'))

    bob.core.log.add_command_line_option(parser)

    # parse command line options
    args = parser.parse_args(command_line_parameters)
    bob.core.log.set_verbosity_level(logger, args.verbose)

    if len(args.system_names) != len(args.system_types):
        logger.error("Please specify a type of system for each given name")
        return

    command_line_parameters = []

    # add training scores
    train_positive_only_scores = ''
    if args.fusion_type == 'cascade' and args.fusion_algorithm != 'and':
        train_positive_only_scores = '-pos'
    command_line_parameters.append('-t')
    for sysname, systype in zip(args.system_names, args.system_types):
        score_path = os.path.join(args.prefix_path, systype, args.dataset_name_train, sysname, 'scores-')
        if args.dataset_name_asv and systype == 'asv':
            score_path = os.path.join(args.prefix_path, systype, args.dataset_name_asv, sysname, 'scores-')
        command_line_parameters.append(score_path + 'train' + train_positive_only_scores)
        train_positive_only_scores = ''  # only first system can have positive only scores

    # add dev and/or eval scores
    for group in args.score_group:
        score_modifier = '-d'
        dataset_name = args.dataset_name_train
        if group == 'eval':
            score_modifier = '-e'
            dataset_name = args.dataset_name_eval
        positive_only_scores = ''
        if args.fusion_type == 'cascade':
            positive_only_scores = '-pos'
        command_line_parameters.append(score_modifier)
        for sysname, systype in zip(args.system_names, args.system_types):
            score_path = os.path.join(args.prefix_path, systype, dataset_name, sysname, 'scores-')
            if args.dataset_name_asv and systype == 'asv':
                score_path = os.path.join(args.prefix_path, systype, args.dataset_name_asv, sysname, 'scores-')
            command_line_parameters.append(score_path + group + positive_only_scores)
            positive_only_scores = ''  # only first system can have positive only scores

    # add the rest of the parameters
    command_line_parameters.extend(['-a', args.fusion_algorithm])
    command_line_parameters.extend(['-T', 'scores-train', '-vvv'])
    # command_line_parameters.extend(['--force', '-vvv'])
    # command_line_parameters.extend(['--filter-scores', '-vvv'])
    out_dir = '-'.join([args.out_directory, args.fusion_type] + args.system_names + [args.dataset_name_eval, args.fusion_algorithm])
    command_line_parameters.extend(['-s', out_dir])

    logger.info("Starting fusion script with the following command line parameters:\n %s", ' '.join(command_line_parameters))
    bob.fusion.base.script.bob_fuse.main(command_line_parameters)

if __name__ == '__main__':
    main()
