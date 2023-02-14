#!/usr/bin/env python

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import bob.fusion.base

algorithm = bob.fusion.base.algorithm.LLR(
    preprocessors=[StandardScaler(), PolynomialFeatures(degree=2)]
)

algorithm_tanh = bob.fusion.base.algorithm.LLR(
    preprocessors=[
        bob.fusion.base.preprocessor.Tanh(),
        PolynomialFeatures(degree=2),
    ]
)
