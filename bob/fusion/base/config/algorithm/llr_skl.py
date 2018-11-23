#!/usr/bin/env python

import bob.fusion.base
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

algorithm = bob.fusion.base.algorithm.Algorithm(
    preprocessors=[StandardScaler()],
    classifier=LogisticRegression())

algorithm_tanh = bob.fusion.base.algorithm.Algorithm(
    preprocessors=[bob.fusion.base.preprocessor.Tanh()],
    classifier=LogisticRegression())
