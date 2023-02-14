import logging

from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def get_2negatives_1positive(score_lines):
    gen_mask = score_lines["claimed_id"] == score_lines["real_id"]
    atk_mask = np.logical_or(
        np.char.count(score_lines["real_id"], "spoof") > 0,
        np.char.count(score_lines["real_id"], "attack") > 0,
    )
    zei_mask = np.logical_and(
        np.logical_not(gen_mask), np.logical_not(atk_mask)
    )
    gen = score_lines[gen_mask]
    zei = score_lines[zei_mask]
    atk = score_lines[atk_mask]
    return (gen, zei, atk, gen_mask, zei_mask, atk_mask)


def check_consistency(gen_l, zei_l, atk_l):
    if len(gen_l) < 2:
        raise ValueError(
            "Check failed since less than two system is available."
        )
    for score_lines_list in (gen_l, zei_l, atk_l):
        if not score_lines_list:
            continue
        score_lines0 = score_lines_list[0]
        for score_lines in score_lines_list[1:]:
            match = np.all(
                score_lines["claimed_id"] == score_lines0["claimed_id"]
            )
            if not match:
                raise ValueError("claimed ids do not match between score files")

            match = np.all(score_lines["real_id"] == score_lines0["real_id"])
            if not match:
                raise ValueError("real ids do not match between score files")


def get_scores(*args):
    scores = []
    for temp in zip(*args):
        scores.append(np.concatenate([a["score"] for a in temp], axis=0))
    return np.vstack(scores).T


def get_score_lines(*args):
    # get the dtype names
    names = list(args[0][0].dtype.names)
    if len(names) != 4:
        names = [n for n in names if "model_label" not in n]
    logger.debug(names)

    # find the (max) size of strigns
    dtypes = [a.dtype for temp in zip(*args) for a in temp]
    lengths = defaultdict(list)
    for name in names:
        for d in dtypes:
            lengths[name].append(d[name].itemsize // 4)

    # make a new dtype
    new_dtype = []
    for name in names[:-1]:
        new_dtype.append((name, "U{}".format(max(lengths[name]))))
    new_dtype.append((names[-1], float))

    score_lines = []
    for temp in zip(*args):
        for a in temp:
            score_lines.extend(a[names].tolist())
    score_lines = np.array(score_lines, dtype=new_dtype)
    return score_lines


def remove_nan(samples, found_nan):
    ncls = samples.shape[1]
    nans = np.isnan(samples[:, 0])
    for i in range(1, ncls):
        nans = np.logical_or(nans, np.isnan(samples[:, i]))
    return np.any(nans) or found_nan, nans, samples[~nans, :]


def get_gza_from_lines_list(score_lines_list):
    gen_l, zei_l, atk_l = [], [], []
    for score_lines in score_lines_list:
        gen, zei, atk, _, _, _ = get_2negatives_1positive(score_lines)
        gen_l.append(gen)
        zei_l.append(zei)
        atk_l.append(atk)
    zei_lengths = []
    for zei in zei_l:
        zei_lengths.append(zei.size)
    zei_lengths = np.array(zei_lengths)
    idx1 = 0  # used later if it does not enter the if.
    if not (np.all(zei_lengths == 0) or np.all(zei_lengths > 0)):
        logger.info(
            "Trying to fill-in the missing zero effort impostor scores"
            " for pad systems. If you see a numpy index error below, "
            "your biometric scores do not match your pad scores."
        )
        # generate the missing ones
        # find one that has zei
        idx1 = zei_lengths.nonzero()[0][0]
        zei_full = zei_l[idx1]
        for idx2 in np.where(zei_lengths == 0)[0]:
            if zei_l[idx2] is None:
                continue
            temp = np.array(zei_full)
            # make sure we replace all scores.
            temp["score"] = np.nan
            # get the list of ids
            real_ids = np.unique(temp["real_id"])
            # find pad score of that id and replace the score
            for real_id in real_ids:
                # get the list of test_labels
                test_labels = np.unique(
                    temp["test_label"][temp["real_id"] == real_id]
                )
                for test_label in test_labels:
                    idx3 = np.logical_and(
                        temp["real_id"] == real_id,
                        temp["test_label"] == test_label,
                    )
                    idx4 = np.logical_and(
                        gen_l[idx2]["real_id"] == real_id,
                        gen_l[idx2]["test_label"] == test_label,
                    )
                    temp["score"][idx3] = gen_l[idx2]["score"][idx4]
            zei_l[idx2] = temp
    return idx1, gen_l, zei_l, atk_l
