#!/usr/bin/env python

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import bob.fusion.base

algorithm = bob.fusion.base.algorithm.Algorithm(
    preprocessors=[StandardScaler()], classifier=LogisticRegression()
)

algorithm_tanh = bob.fusion.base.algorithm.Algorithm(
    preprocessors=[bob.fusion.base.preprocessor.Tanh()],
    classifier=LogisticRegression(),
)
