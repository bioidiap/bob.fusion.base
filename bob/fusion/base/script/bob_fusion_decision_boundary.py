#!/usr/bin/env python

"""Plot decision boundraries of the fusion algorithm."""

import argparse
import numpy as np

import bob.fusion.base
import bob.core
from bob.measure.load import load_score
from ..tools import grouping, get_gza_from_lines_list, \
    get_scores, remove_nan, check_consistency
logger = bob.core.log.setup("bob.fusion.base")


def plot_boundary_decision(algorithm, scores, score_labels, threshold,
                           thres_system1=None,
                           thres_system2=None,
                           do_grouping=False,
                           resolution=2000,
                           # x_pad=0.5,
                           # y_pad=0.5,
                           alpha=0.75,
                           legends=None,
                           i1=0,
                           i2=1,
                           **kwargs
                           ):
    '''
    Plots the boundary decision of the Algorithm

    @param score_labels np.array A (scores.shape[0]) array containing
                                                                    the true labels of scores.

    @param threshold    float       threshold of the decision boundary
    '''
    if legends is None:
        legends = ['Zero Effort Impostor', 'Presentation Attack', 'Genuine']
    markers = ['x', 'o', 's']

    if scores.shape[1] > 2:
        raise NotImplementedError(
            "Currently plotting the decision boundary for more than two systems "
            "is not supported.")

    import matplotlib.pyplot as plt
    plt.gca()  # this is necessary for subplots to work.

    X = scores[:, [i1, i2]]
    Y = score_labels
    x_pad = (X[:, i1].max() - X[:, i1].min()) * 0.1
    y_pad = (X[:, i2].max() - X[:, i2].min()) * 0.1
    x_min, x_max = X[:, i1].min() - x_pad, X[:, i1].max() + x_pad
    y_min, y_max = X[:, i2].min() - y_pad, X[:, i2].max() + y_pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution))
    temp = np.c_[xx.ravel(), yy.ravel()]
    temp = algorithm.preprocess(temp)
    Z = (algorithm.fuse(temp) > threshold).reshape(xx.shape)

    contourf = plt.contour(xx, yy, Z, 1, alpha=1, cmap=plt.cm.gray)

    if do_grouping:
        gen = grouping(X[Y == 0, :], **kwargs)
        zei = grouping(X[Y == 1, :], **kwargs)
        atk = grouping(X[Y == 2, :], **kwargs)
    else:
        gen = X[Y == 0, :]
        zei = X[Y == 1, :]
        atk = X[Y == 2, :]
    colors = plt.cm.viridis(np.linspace(0, 1, 3))
    for i, X in enumerate((zei, atk, gen)):
        plt.scatter(
            X[:, 0], X[:, 1], marker=markers[i], alpha=alpha,
            c=colors[i], label=legends[i])
    # plt.legend(loc='best')
    plt.legend(bbox_to_anchor=(-0.05, 1.02, 1.05, .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0., fontsize=14)

    if thres_system1 is not None:
        plt.axvline(thres_system1, color='red')
        plt.axhline(thres_system2, color='red')

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.grid('on')

    plt.xlabel('Face recognition scores')
    plt.ylabel('PAD scores')

    return contourf


def main(command_line_parameters=None):

    # setup command line parameters
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--eval-files', nargs='+', required=True,
                        help='A list of score files to be plotted usually the '
                        'evaluation set.')

    parser.add_argument('-m', '--model-file', required=True,
                        help='Path to Model.pkl of a saved bob.fusion.algorithm.')

    parser.add_argument('-t', '--threshold', type=float, default=0, required=True,
                        help='The threshold to classify scores after fusion.'
                        'Usually calculated from fused development set.')

    parser.add_argument('-o', '--output', default='scatter.pdf',
                        help='The path to save the plot.')

    parser.add_argument('-g', '--group', default=0, type=int,
                        help='If given scores will be grouped into N samples.')

    parser.add_argument('-G', '--grouping', choices=('random', 'kmeans'),
                        default='kmeans',
                        help='The gouping algorithm to be used.')

    parser.add_argument('--skip-check', action='store_true',
                        help='If provided, score files are not checked '
                        'for consistency')

    parser.add_argument('--score-type', choices=(4, 5), default=None,
                        help='The format the scores are provided.')

    bob.core.log.add_command_line_option(parser)

    # parse command line options
    args = parser.parse_args(command_line_parameters)
    bob.core.log.set_verbosity_level(logger, args.verbose)

    # load the algorithm
    algorithm = bob.fusion.base.algorithm.Algorithm()
    algorithm = algorithm.load(args.model_file)

    # load the scores
    score_lines_list_eval = [load_score(path, ncolumns=args.score_type)
                             for path in args.eval_files]

    # genuine, zero effort impostor, and attack list
    idx1, gen_le, zei_le, atk_le = get_gza_from_lines_list(score_lines_list_eval)

    # check if score lines are consistent
    if not args.skip_check:
        check_consistency(gen_le, zei_le, atk_le)

    # concatenate the scores and create the labels
    scores = get_scores(gen_le, zei_le, atk_le)
    score_labels = np.zeros((scores.shape[0],))
    gensize = gen_le[0].shape[0]
    zeisize = zei_le[0].shape[0]
    score_labels[:gensize] = 0
    score_labels[gensize: gensize + zeisize] = 1
    score_labels[gensize + zeisize:] = 2
    found_nan, nan_idx, scores = remove_nan(scores, False)
    score_labels = score_labels[~nan_idx]

    if found_nan:
        logger.warn('{} nan values were removed.'.format(np.sum(nan_idx)))

    # plot the decision boundary
    do_grouping = True
    if args.group < 1:
        do_grouping = False

    import matplotlib
    if not hasattr(matplotlib, 'backends'):
        matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    plot_boundary_decision(
        algorithm, scores, score_labels, args.threshold,
        do_grouping=do_grouping,
        npoints=args.group,
        seed=0,
        gformat=args.grouping
    )
    plt.savefig(args.output, transparent=True)
    plt.close()


if __name__ == '__main__':
    main()
