"""Plots the decision boundaries of fusion algorithms.
"""
import logging

import click
import numpy as np

from clapper.click import verbosity_option

from bob.bio.base.score import load_score

from ..algorithm import Algorithm
from ..tools import (
    check_consistency,
    get_gza_from_lines_list,
    get_scores,
    grouping,
    remove_nan,
)

logger = logging.getLogger(__name__)


def plot_boundary_decision(
    algorithm,
    scores,
    score_labels,
    threshold,
    thres_system1=None,
    thres_system2=None,
    do_grouping=False,
    resolution=2000,
    alpha=0.75,
    legends=None,
    i1=0,
    i2=1,
    x_label=None,
    y_label=None,
    **kwargs,
):
    if legends is None:
        legends = ["Zero Effort Impostor", "Presentation Attack", "Genuine"]
    markers = ["x", "o", "s"]

    if scores.shape[1] > 2:
        raise NotImplementedError(
            "Currently plotting the decision boundary for more than two "
            "systems is not supported."
        )

    import matplotlib
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
        np.linspace(y_min, y_max, resolution),
    )

    contourf = None
    if algorithm is not None:
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
    for i, (X, color) in enumerate(((zei, "C0"), (atk, "C1"), (gen, "C2"))):
        if X.size == 0:
            continue
        try:
            plt.scatter(
                X[:, 0],
                X[:, 1],
                marker=markers[i],
                alpha=alpha,
                c=color,
                label=legends[i],
            )
        except Exception as e:
            raise RuntimeError(
                f"matplotlib backend: {matplotlib.get_backend()}"
            ) from e

    plt.legend(
        bbox_to_anchor=(-0.05, 1.02, 1.05, 0.102),
        loc=3,
        ncol=3,
        mode="expand",
        borderaxespad=0.0,
        fontsize=14,
    )

    if thres_system1 is not None:
        plt.axvline(thres_system1, color="red")
        plt.axhline(thres_system2, color="red")

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.grid(True)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    return contourf


@click.command(
    epilog="""\b
Examples:
$ bob fusion boundary -vvv {sys1,sys2}/scores-eval -m /path/to/Model.pkl
"""
)
@click.argument("scores", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "-m",
    "--model-file",
    required=False,
    help="The path to where the algorithm will be loaded from.",
)
@click.option(
    "-t",
    "--threshold",
    type=click.FLOAT,
    required=False,
    help="The threshold to classify scores after fusion. Usually "
    "calculated from fused development set.",
)
@click.option(
    "-g",
    "--group",
    type=click.INT,
    default=0,
    show_default=True,
    help="If given scores will be grouped into N samples.",
)
@click.option(
    "-G",
    "--grouping",
    type=click.Choice(("random", "kmeans")),
    default="kmeans",
    show_default=True,
    help="The gouping algorithm to be used.",
)
@click.option(
    "-o",
    "--output",
    default="scatter.pdf",
    show_default=True,
    type=click.Path(writable=True),
    help="The path to the saved plot.",
)
@click.option(
    "-X",
    "--x-label",
    default="Recognition scores",
    show_default=True,
    help="The label for the first system.",
)
@click.option(
    "-Y",
    "--y-label",
    default="PAD scores",
    show_default=True,
    help="The label for the second system.",
)
@click.option(
    "--skip-check",
    is_flag=True,
    show_default=True,
    help="If True, it will skip checking for the consistency "
    "between scores.",
)
@verbosity_option(logger)
def boundary(
    scores,
    model_file,
    threshold,
    group,
    grouping,
    output,
    x_label,
    y_label,
    skip_check,
    **kwargs,
):
    """Plots the decision boundaries of fusion algorithms.

    The script takes several scores (usually eval scores) from different
    biometric and pad systems and a trained algorithm and plots the decision
    boundary.

    You need to provide two score files from two systems. System 1 will be
    plotted on the x-axis.
    """
    # load the algorithm
    algorithm = None
    if model_file:
        algorithm = Algorithm().load(model_file)
        assert (
            threshold is not None
        ), "threshold must be provided with the model"

    # load the scores
    score_lines_list_eval = [load_score(path) for path in scores]

    # genuine, zero effort impostor, and attack list
    idx1, gen_le, zei_le, atk_le = get_gza_from_lines_list(
        score_lines_list_eval
    )

    # check if score lines are consistent
    if not skip_check:
        check_consistency(gen_le, zei_le, atk_le)

    # concatenate the scores and create the labels
    scores = get_scores(gen_le, zei_le, atk_le)
    score_labels = np.zeros((scores.shape[0],))
    gensize = gen_le[0].shape[0]
    zeisize = zei_le[0].shape[0]
    score_labels[:gensize] = 0
    score_labels[gensize : gensize + zeisize] = 1
    score_labels[gensize + zeisize :] = 2
    found_nan, nan_idx, scores = remove_nan(scores, False)
    score_labels = score_labels[~nan_idx]

    if found_nan:
        logger.warn("{} nan values were removed.".format(np.sum(nan_idx)))

    # plot the decision boundary
    do_grouping = True
    if group < 1:
        do_grouping = False

    import matplotlib.pyplot as plt

    plot_boundary_decision(
        algorithm,
        scores,
        score_labels,
        threshold,
        do_grouping=do_grouping,
        npoints=group,
        seed=0,
        gformat=grouping,
        x_label=x_label,
        y_label=y_label,
    )
    plt.savefig(output, transparent=True)
    plt.close()
