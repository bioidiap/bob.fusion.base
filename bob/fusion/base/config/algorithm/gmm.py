#!/usr/bin/env python

import bob.fusion.base
from sklearn.preprocessing import StandardScaler

algorithm = bob.fusion.base.algorithm.GMM(preprocessors=[StandardScaler()])

algorithm_tanh = bob.fusion.base.algorithm.GMM(
    preprocessors=[bob.fusion.base.preprocessor.Tanh()])
