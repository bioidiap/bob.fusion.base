#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler

import bob.fusion.base

algorithm = bob.fusion.base.algorithm.Weighted_Sum(
    preprocessors=[StandardScaler()]
)

algorithm_tanh = bob.fusion.base.algorithm.Weighted_Sum(
    preprocessors=[bob.fusion.base.preprocessor.Tanh()]
)
