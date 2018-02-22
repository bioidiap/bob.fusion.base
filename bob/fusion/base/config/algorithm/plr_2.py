#!/usr/bin/env python

import bob.fusion.base
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

algorithm = bob.fusion.base.algorithm.Algorithm(
  preprocessors=[StandardScaler(), PolynomialFeatures(degree=2)],
  classifier=LogisticRegression())
