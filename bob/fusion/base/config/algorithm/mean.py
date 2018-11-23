#!/usr/bin/env python

import bob.fusion.base
from sklearn.preprocessing import StandardScaler

algorithm = bob.fusion.base.algorithm.Weighted_Sum(
    preprocessors=[StandardScaler()])

algorithm_tanh = bob.fusion.base.algorithm.Weighted_Sum(
    preprocessors=[bob.fusion.base.preprocessor.Tanh()])
