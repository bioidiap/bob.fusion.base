#!/usr/bin/env python

import bob.fusion.base
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

algorithm = bob.fusion.base.algorithm.LLR(
  preprocessors=[StandardScaler(), PolynomialFeatures(degree=2)])
