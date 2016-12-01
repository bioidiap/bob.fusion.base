#!/usr/bin/env python

"""Score Fusion script
"""

import argparse
import os
import sys
import pkg_resources
from bob.bio.base import tools, utils
from bob.io.base import create_directories_safe

import bob.core
logger = bob.core.log.setup("bob.bio.base")
valid_keywords = ('algorithm')


def _get_entry_points(keyword, strip=[]):
  """Returns the list of entry points for registered resources with the given
  keyword."""
  return [entry_point for entry_point in
          pkg_resources.iter_entry_points('bob.fusion.' + keyword)
          if not entry_point.name.startswith(tuple(strip))]


def resource_keys(keyword, exclude_packages=[], strip=['dummy']):
  """Reads and returns all resources that are registered with the given keyword.
  Entry points from the given ``exclude_packages`` are ignored."""
  return sorted([entry_point.name for entry_point in
                 _get_entry_points(keyword, strip) if
                 entry_point.dist.project_name not in exclude_packages])


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
  parser.add_argument('-o', '--fused-dev-file', default=None,
                      help='The fused development score file. '
                           'Default is "scores-dev" in the --save-directory')
  parser.add_argument('-O', '--fused-eval-file', default=None,
                      help='The fused evaluation score file. '
                           'Default is "scores-eval" in the --save-directory')
  parser.add_argument('-T', '--fused-train-file', default=None,
                      help='The fused training score file. '
                           'Default is "scores-train" in the --save-directory')
  parser.add_argument('--score-type', choices=[4, 5], default=None,
                      help='The format the scores are provided. If not '
                           'provided, the number of columns will be guessed.')
  parser.add_argument('--skip-check', action='store_true',
                      help='If provided, score files are not checked '
                           'for consistency')
  parser.add_argument('--filter-scores', action='store_true',
                      help='If provided, score files are filtered across all systems to the smallest common subset '
                           'of files.')
  parser.add_argument('-s', '--save-directory', help='The directory to save '
                      'the experiment artifacts.', default='fusion_result')

  config_group = parser.add_argument_group(
    'Parameters defining the experiment', ' Most of these parameters can be a'
    ' registered resource, a configuration file, or even a string that '
    'defines a newly created object')
  config_group.add_argument(
    '-a', '--algorithm', metavar='x', required=True,
    help='Fusion; registered algorithms are: %s' % resource_keys(
      'algorithm', exclude_resources_from))
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
  args.algorithm = load_resource(
    args.algorithm, 'algorithm', imports=args.imports)

  # set base directories
  if args.fused_train_file is not None:
    args.fused_train_file = os.path.join(args.save_directory, args.fused_train_file)
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


def load_resource(resource, keyword, imports=['bob.fusion.base'],
                  preferred_package=None):
  """Loads the given resource that is registered with the given keyword.
  The resource can be:

  1. a resource as defined in the setup.py
  2. a configuration file
  3. a string defining the construction of an object. If imports are required
     for the construction of this object, they can be given as list of strings.

  **Parameters:**

  resource : str
    Any string interpretable as a resource (see above).

  keyword : str
    A valid resource keyword, can be one of :py:attr:`valid_keywords`.

  imports : [str]
    A list of strings defining which modules to import, when constructing new
      objects (option 3).

  preferred_package : str or ``None``
    When several resources with the same name are found in different packages
      (e.g., in different ``bob.bio`` or other packages), this specifies the
      preferred package to load the resource from. If not specified, the
      extension that is **not** from ``bob.bio`` is selected.

  **Returns:**

  resource : object
    The resulting resource object is returned, either read from file or
      resource, or created newly.
  """

  # first, look if the resource is a file name
  if os.path.isfile(resource):
    return utils.read_config_file(resource, keyword)

  if keyword not in valid_keywords:
    raise ValueError("The given keyword '%s' is not valid. "
                     "Please use one of %s!" % (str(keyword),
                                                str(valid_keywords)))

  # now, we check if the resource is registered as an entry point in the
  # resource files
  entry_points = [entry_point for entry_point in _get_entry_points(
    keyword) if entry_point.name == resource]

  if len(entry_points):
    if len(entry_points) == 1:
      return entry_points[0].load()
    else:
      # TODO: extract current package name and use this one, if possible

      # Now: check if there are only two entry points, and one is from the
      # bob.fusion.base, then use the other one
      index = -1
      if preferred_package is not None:
        for i, p in enumerate(entry_points):
          if p.dist.project_name == preferred_package:
            index = i
            break

      if index == -1:
        # by default, use the first one that is not from bob.bio
        for i, p in enumerate(entry_points):
          if not p.dist.project_name.startswith('bob.bio'):
            index = i
            break

      if index != -1:
        logger.debug("RESOURCES: Using the resource '%s' from '%s', "
                     "and ignoring the one from '%s'",
                     resource, entry_points[index].module_name,
                     entry_points[1 - index].module_name)
        return entry_points[index].load()
      else:
        logger.warn("Under the desired name '%s', there are multiple "
                    "entry points defined, we return the first one: %s",
                    resource,
                    [entry_point.module_name for entry_point in entry_points])
        return entry_points[0].load()

  # if the resource is neither a config file nor an entry point,
  # just execute it as a command
  try:
    # first, execute all import commands that are required
    for i in imports:
      exec("import %s" % i)
    # now, evaluate the resource (re-evaluate if the resource is still a
    # string)
    while isinstance(resource, str):
      resource = eval(resource)
    return resource

  except Exception as e:
    raise ImportError("The given command line option '%s' is neither a "
                      "resource for a '%s', nor an existing configuration"
                      " file, nor could be interpreted as a command "
                      "(error: %s)" % (resource, keyword, str(e)))
