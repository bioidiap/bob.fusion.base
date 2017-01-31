#!/usr/bin/env python

"""Score Fusion script
"""

import argparse
import os
import sys
from bob.bio.base import tools, utils
from bob.io.base import create_directories_safe

import bob.core
logger = bob.core.log.setup("bob.bio.base")
valid_keywords = ('algorithm')


def command_line_parser(description=__doc__, exclude_resources_from=[]):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--train-files', required=True,
                        nargs='+', help="A list of score files of "
                        "the train set.")
    parser.add_argument('-d', '--dev-files', nargs='+',
                        help="A list of score files of the development set; "
                        "if given it must be the same number of files "
                        "as the --train-files.")
    parser.add_argument('-e', '--eval-files', nargs='+',
                        help="A list of score files of the evaluation set; "
                        "if given it must be the same number of files "
                        "as the --train-files.")
    parser.add_argument('-T', '--fused-train-file', default=None,
                        help='The fused train score file. '
                        'Default is "scores-train" in the --save-directory')
    parser.add_argument('-D', '--fused-dev-file', default=None,
                        help='The fused development score file. '
                        'Default is "scores-dev" in the --save-directory')
    parser.add_argument('-E', '--fused-eval-file', default=None,
                        help='The fused evaluation score file. '
                        'Default is "scores-eval" in the --save-directory')
    parser.add_argument('--score-type', choices=[4, 5], default=None,
                        help='The format the scores are provided. If not '
                        'provided, the number of columns will be guessed.')
    parser.add_argument('--skip-check', action='store_true',
                        help='If provided, score files are not checked '
                        'for consistency')
    parser.add_argument('-s', '--save-directory', help='The directory to save '
                        'the experiment artifacts.', default='fusion_result')

    config_group = parser.add_argument_group(
        'Parameters defining the experiment', ' Most of these parameters can be a'
        ' registered resource, a configuration file, or even a string that '
        'defines a newly created object')
    config_group.add_argument(
        '-a', '--algorithm', metavar='x', required=True,
        help='Fusion; registered algorithms are: %s' % utils.resource_keys(
            'algorithm', exclude_resources_from, package_prefix='bob.fusio.'))
    config_group.add_argument(
        '-m', '--imports', metavar='LIB', nargs='+',
        default=['bob.fusion.base'], help='If one of your configuration files is'
        ' an actual command, please specify the lists of'
        ' required libraries (imports) to execute this command')

    flag_group = parser.add_argument_group(
        'Flags that change the behavior of the experiment')
    bob.core.log.add_command_line_option(flag_group)
    flag_group.add_argument('-F', '--force', action='store_true',
                                                    help='Force to erase former data if already exist')

    return {
        'main': parser,
        'config': config_group,
        'flag': flag_group
    }


def initialize(parsers, command_line_parameters=None, skips=[]):

    args = parsers['main'].parse_args(command_line_parameters)

    # logging
    bob.core.log.set_verbosity_level(logger, args.verbose)

    # load configuration resources
    args.algorithm = utils.load_resource(
        args.algorithm, 'algorithm', imports=args.imports,
        package_prefix='bob.fusion.')

    # set base directories
    if args.fused_train_file is None:
        args.fused_train_file = os.path.join(args.save_directory, 'scores-train')
    if args.fused_dev_file is None:
        args.fused_dev_file = os.path.join(args.save_directory, 'scores-dev')
    if args.fused_eval_file is None:
        args.fused_eval_file = os.path.join(args.save_directory, 'scores-eval')

    # result files
    args.info_file = os.path.join(args.save_directory, 'Experiment.info')

    args.model_file = os.path.join(args.save_directory, 'Model.pkl')

    return args


def write_info(args, command_line_parameters):
    """Writes information about the current experimental setup into a file
    specified on command line.

    **Parameters:**

    args : namespace
        The interpreted command line arguments as returned by the
            :py:func:`initialize` function.

    command_line_parameters : [str] or ``None``
        The command line parameters that have been interpreted.
        If ``None``, the parameters specified by the user on command line
        are considered.

    executable : str
        The name of the executable (such as ``'./bin/verify.py'``) that is used
            to run the experiments.
    """
    if command_line_parameters is None:
        command_line_parameters = sys.argv[1:]
    executable = sys.argv[0]
    # write configuration
    try:
        create_directories_safe(os.path.dirname(args.info_file))
        with open(args.info_file, 'w') as f:
            f.write("Command line:\n")
            f.write(
                tools.command_line([executable] + command_line_parameters) + "\n\n")
            f.write("Configuration:\n\n")
            f.write("Algorithm:\n%s\n\n" % args.algorithm)
    except IOError:
        logger.error(
            "Could not write the experimental setup into file '%s'", args.info_file)


def parse_arguments(command_line_parameters, exclude_resources_from=[]):
    """This function parses the given options (which by default are the command
        line options). If exclude_resources_from is specified (as a list), the
        resources from the given packages are not listed in the help message."""
    # set up command line parser
    parsers = command_line_parser(exclude_resources_from=exclude_resources_from)

    # now that we have set up everything, get the command line arguments
    return initialize(parsers, command_line_parameters)
