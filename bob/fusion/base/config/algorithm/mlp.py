#!/usr/bin/env python

import bob.fusion.base
from sklearn.preprocessing import StandardScaler

algorithm = bob.fusion.base.algorithm.MLP(
    preprocessors=[StandardScaler()])

algorithm_tanh = bob.fusion.base.algorithm.MLP(
    preprocessors=[bob.fusion.base.preprocessor.Tanh()])
