#!/usr/bin/env python

import bob.fusion.base
import sklearn.preprocessing

algorithm = bob.fusion.base.algorithm.LogisticRegression(
  preprocessors=[(sklearn.preprocessing.RobustScaler(), False),
                 (sklearn.preprocessing.PolynomialFeatures(degree=2), False)])
