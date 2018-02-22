#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import numpy

from .Algorithm import Algorithm

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


class CascadeFuse(Algorithm):
  """weighted sum (default: mean)"""

  def __init__(self, *args, **kwargs):
    super(CascadeFuse, self).__init__(
      classifier=self,
      *args, **kwargs)

    self.thres_dev = 0.0

  def train(self, train_neg, train_pos, devel_neg=None, devel_pos=None):
    super(CascadeFuse, self).train(train_neg, train_pos, devel_neg=None, devel_pos=None)

    # compute threshold from the development set scores of the second system (column 1)
    from bob.measure import eer_threshold
    self.thres_dev = eer_threshold(devel_neg[:, 1], devel_pos[:, 1])
    print (self.thres_dev)
    # print ("shape dev_neg: ", devel_neg.shape)
    # print (devel_neg[1])
    # print ("shape dev_pos: ", devel_pos.shape)
    # print (devel_pos[1])

  def fit(self, X, y):
    # no training for this approach
    pass

  def decision_function(self, scores):
    # we assume that scores in 0st column are used to decide outcome
    # we operate on the scores from the 1st column, as this is the system that is the second in cascade
    # compute threshold for the second system (column 1)
    # if scores of the first system (column 0) are less then 0 (threshold of the first system)
    # we shift the scores of the second system by its own threshold (so that they become less than threshold)

    # shift the scores that did not pass through the first system by the devel threshold of the second
    # idialy, we need to shift by the threshold of the eval set, but since we do not know it,
    #
    print ("pre-computed threshold:", self.thres_dev)
    return [score_set[1]-self.thres_dev if score_set[0] < 0 else score_set[1] for score_set in scores]
    # return [score_set[0] if score_set[0] < 0 else score_set[1] for score_set in scores]
    # return [score_set[0] if score_set[0] < 0.2 else score_set[1] for score_set in scores]
    # return [score_set[0] if score_set[0] == numpy.finfo(numpy.float16).min else score_set[1] for score_set in scores]
