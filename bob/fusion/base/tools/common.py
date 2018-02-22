import numpy as np

import bob.core

logger = bob.core.log.setup("bob.fusion.base")


def get_2negatives_1positive(score_lines):
    gen_mask = score_lines['claimed_id'] == score_lines['real_id']
    check_attacks = np.array([False if len(a) == 3 else True for a in score_lines['real_id']])
    atk_mask = np.logical_or(np.char.count(score_lines['real_id'], 'spoof/') > 0,
                             np.char.count(score_lines['real_id'], 'attack') > 0,
                             check_attacks)
    zei_mask = np.logical_and(np.logical_not(gen_mask), np.logical_not(atk_mask))
    gen = score_lines[gen_mask]
    zei = score_lines[zei_mask]
    atk = score_lines[atk_mask]
    return (gen, zei, atk, gen_mask, zei_mask, atk_mask)


def filter_to_common_scores(gen_l, zei_l, atk_l):
    from functools import reduce
    if len(gen_l) < 2:
        logger.error('Check failed since less than two system is available.')
    for score_lines_list in gen_l, zei_l, atk_l:
        # remove score column
        new_dtype = score_lines_list[0].dtype
        new_dtype = [x for x in eval(str(new_dtype)) if not x[0] == 'score']
        # find the largest common subset of rows among all systems
        common = reduce(np.intersect1d, [np.array(x, dtype=new_dtype) for x in score_lines_list])
        if len(common) == 0:
            raise ValueError('Zero intersection found between scores files')

        # contract the scores
        # bring them to the common denominator
        for i, score_lines in enumerate(score_lines_list):
            mask = np.in1d(np.array(score_lines, dtype=new_dtype), common)
            # keep only those rows that are inside the largest common subset
            score_lines_list[i] = score_lines[mask]

def check_consistency(gen_l, zei_l, atk_l):
    if len(gen_l) < 2:
        logger.error('Check failed since less than two system is available.')
    for score_lines_list in (gen_l, zei_l, atk_l):
        if not score_lines_list:
            continue
        score_lines0 = score_lines_list[0]
        for score_lines in score_lines_list[1:]:
            # print(len(score_lines['claimed_id']))
            # print(len(score_lines0['claimed_id']))
            # print(score_lines0['claimed_id'])
            # print(score_lines['claimed_id'])
            assert (np.all(score_lines['claimed_id'] == score_lines0['claimed_id']))
            assert (np.all(score_lines['real_id'] == score_lines0['real_id']))


def get_scores(*args):
    scores = []
    for temp in zip(*args):
        scores.append(np.concatenate([a['score'] for a in temp], axis=0))
    return np.vstack(scores).T


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
        # generate the missing ones
        # find one that has zei
        idx1 = zei_lengths.nonzero()[0][0]
        zei_full = zei_l[idx1]
        for idx2 in np.where(zei_lengths == 0)[0]:
            if zei_l[idx2] is None:
                continue
            temp = np.array(zei_full)
            # make sure we replace all scores.
            temp['score'] = np.nan
            # get the list of ids
            real_ids = np.unique(temp['real_id'])
            # find pad score of that id and replace the score
            for real_id in real_ids:
                # get the list of test_labels
                test_labels = np.unique(temp['test_label'][temp['real_id'] == real_id])
                for test_label in test_labels:
                    idx3 = np.logical_and(temp['real_id'] == real_id,
                                          temp['test_label'] == test_label)
                    idx4 = np.logical_and(gen_l[idx2]['real_id'] == real_id,
                                          gen_l[idx2]['test_label'] == test_label)
                    # sometimes an PAD does not have scores corresponding to files from ASV zei
                    # if np.any(idx3) and np.any(idx4):
                    try:
                        temp['score'][idx3] = gen_l[idx2]['score'][idx4]
                    except ValueError:
                        raise ValueError("Trying to assign gen_l[idx2]['score'][idx4]=%s to %s, real_id=%s, test_label=%s" %
                                         (str(gen_l[idx2]['score'][idx4]), str(temp['score'][idx3]), str(real_id), str(test_label)))
            # remove NaN when PAD zie < ASV zei
            # nans = np.isnan(temp['score'])
            # temp = temp[~nans]
            # assign found zei scores for the PAD system
            zei_l[idx2] = temp
    return idx1, gen_l, zei_l, atk_l
