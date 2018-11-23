#!/usr/bin/env python

import bob.fusion.base
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

algorithm = bob.fusion.base.algorithm.LLR(
    preprocessors=[StandardScaler(), PolynomialFeatures(degree=3)])

algorithm_tanh = bob.fusion.base.algorithm.LLR(
    preprocessors=[bob.fusion.base.preprocessor.Tanh(),
                   PolynomialFeatures(degree=3)])
