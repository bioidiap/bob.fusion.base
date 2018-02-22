#!/usr/bin/env python

"""Plot decision boundraries of the fusion algorithm."""

import argparse
# import matplotlib.pyplot as plt
import numpy as np

import bob.fusion.base
import bob.core
from bob.measure.load import load_score
from ..tools import grouping, get_gza_from_lines_list, \
    get_scores, remove_nan, check_consistency, filter_to_common_scores
import os
import numpy
import matplotlib.pyplot as mpl
from matplotlib.lines import Line2D

logger = bob.core.log.setup("bob.fusion.base")

def draw_line(machine, devel_thres, fig):
    n_points = 1000
    xlim = mpl.xlim()
    x = [xlim[0] + (xlim[1] - xlim[0]) / 3.0, xlim[1] - (xlim[1] - xlim[0]) / 3]
    y = mpl.ylim()
    yrange = numpy.arange(y[0], y[1], (y[1] - y[0]) / n_points)
    yrange = numpy.reshape(yrange, [len(yrange), 1])
    x1 = numpy.array([x[0]] * yrange.size);
    x1 = numpy.reshape(x1, [x1.size, 1])
    x2 = numpy.array([x[1]] * yrange.size);
    x2 = numpy.reshape(x2, [x2.size, 1])

    xy1 = numpy.append(x1, yrange, axis=1);
    xy2 = numpy.append(x2, yrange, axis=1)

    yrangex1 = float(devel_thres) - machine(xy1)
    yrangex2 = float(devel_thres) - machine(xy2)
    y = [yrange[numpy.where(numpy.abs(yrangex1) == numpy.min(numpy.abs(yrangex1)))],
         yrange[numpy.where(numpy.abs(yrangex2) == numpy.min(numpy.abs(yrangex2)))]]

    ylim = (numpy.array(xlim) - x[0]) * (y[1] - y[0]) / (x[1] - x[0]) + y[
        0]  # calculating full line (that spans the full plot)

    ax = fig.axes
    # line = Line2D(x, y, color='black', label = 'decision boundary', linewidth=2)
    line = Line2D(xlim, ylim, color='black', label='Decision boundary', linewidth=2)
    ax[0].add_line(line)


def plot_scatter(algorithm, scores, score_labels, threshold, i1 = 0, i2 = 1):

    import matplotlib;
    # matplotlib.use('pdf')  # avoids TkInter threaded start
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.font_manager as fm

    # figure out scores first
    X = scores[:, [i1, i2]]
    Y = score_labels
    gen = X[Y == 0, :]
    zei = X[Y == 1, :]
    atk = X[Y == 2, :]

    pp = PdfPages(os.path.join('scores-scatter.pdf'))
    fig = mpl.figure()
    mpl.rcParams.update({'font.size': 18})
    # because of the large number of samples, we plot only each 100th sample
    imp_range = range(0, zei.shape[0], 100)
    att_range = range(0, atk.shape[0], 100)
    racc_range = range(0, gen.shape[0], 100)

    # color_scheme = {'genuine':'#7bd425', 'impostors':'#257bd4', 'spoofs':'black', 'line':'#d4257b'} #7e25d4
    color_scheme = {'genuine': 'blue', 'impostors': 'red', 'spoofs': 'black', 'line': 'green'}
    linecolor_scheme = {'line1': '#257bd4', 'line2': '#7bd425'}
    linestyle_scheme = {'line1': '-', 'line2': '-'}
    # linecolor_scheme = {'line1': 'blue', 'line2': 'red'}
    alpha_scheme = {'genuine': 0.6, 'impostors': 0.8, 'spoofs': 0.4}
    # alpha_scheme = {'genuine':0.9, 'impostors':0.8, 'spoofs':0.3}

    mpl.plot(zei[imp_range, 0], zei[imp_range, 1], '^', color=color_scheme['impostors'],
             label='Zero-effort Impostors', alpha=alpha_scheme['impostors'])
    mpl.plot(atk[att_range, 0], atk[att_range, 1], 's', color=color_scheme['spoofs'],
             label='Presentation Attacks', alpha=alpha_scheme['spoofs'])
    mpl.plot(gen[racc_range, 0], gen[racc_range, 1], 'o', color=color_scheme['genuine'],
             label='Genuine Users', alpha=alpha_scheme['genuine'])  # alpha = 0.2
    draw_line(algorithm, threshold, fig)
    mpl.legend(prop=fm.FontProperties(size=16), loc=3)  # loc=1

    # plt.xlim([-6, 1.5])
    # plt.ylim([-10, 1.5])

    mpl.xlabel('PAD scores')
    # mpl.ylabel('PAD2 scores')
    mpl.ylabel('Verification scores')
    mpl.grid()
    pp.savefig()

    pp.close()


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

    @param score_labels np.array A (scores.shape[0]) array containing
                                    the true labels of scores.

    @param threshold    float       threshold of the decision boundary
    '''
    if legends is None:
        legends = ['Impostor', 'Attack', 'Genuine']
    markers = ['x', 'o', 's']

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
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution))
    temp = np.c_[xx.ravel(), yy.ravel()]
    temp = algorithm.preprocess(temp)
    Z = (algorithm.fuse(temp) > threshold).reshape(xx.shape)

    contourf = plt.contourf(xx, yy, Z, 1, alpha=1, cmap=plt.cm.viridis)

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
    plt.legend(loc=4)

    if thres_system1 is not None:
        plt.axvline(thres_system1, color='red')
        plt.axhline(thres_system2, color='red')

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.grid('on')

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
    # algorithm = bob.fusion.base.algorithm.Algorithm()
    # algorithm = algorithm.load(args.model_file)

    hdf5_path = os.path.splitext(args.model_file)[0] + '.hdf5'
    machine = None
    if os.path.isfile(hdf5_path):
        hdf5file = bob.io.base.HDF5File(hdf5_path)
        machine = bob.learn.linear.Machine(hdf5file)

    # load the scores
    score_lines_list_eval = [load_score(path, ncolumns=args.score_type)
                             for path in args.eval_files]

    # genuine, zero effort impostor, and attack list
    idx1, gen_le, zei_le, atk_le = get_gza_from_lines_list(score_lines_list_eval)

    filter_to_common_scores(gen_le, zei_le, atk_le)

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

    # plot_boundary_decision(
    #     algorithm, scores, score_labels, args.threshold,
    #     do_grouping=do_grouping,
    #     npoints=args.group,
    #     seed=0,
    #     gformat=args.grouping
    # )
    # plt.savefig(args.output)
    # plt.close()

    if machine is not None:
        plot_scatter(machine, scores, score_labels, args.threshold)

if __name__ == '__main__':
    main()
