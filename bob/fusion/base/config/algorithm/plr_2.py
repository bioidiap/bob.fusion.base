#!/usr/bin/env python

import bob.fusion.base
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from bob.fusion.base.preprocessor import LLRCalibration

algorithm = bob.fusion.base.algorithm.Algorithm(
  preprocessors=[StandardScaler(), PolynomialFeatures(degree=2)],
  # preprocessors=[LLRCalibration(), PolynomialFeatures(degree=2)],
  classifier=LogisticRegression())

# algorithm = bob.fusion.base.algorithm.LLR(
#   preprocessors=[LLRCalibration(), PolynomialFeatures(degree=2)])
