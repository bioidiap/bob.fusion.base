#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler

import bob.fusion.base

algorithm = bob.fusion.base.algorithm.GMM(preprocessors=[StandardScaler()])

algorithm_tanh = bob.fusion.base.algorithm.GMM(
    preprocessors=[bob.fusion.base.preprocessor.Tanh()]
)
