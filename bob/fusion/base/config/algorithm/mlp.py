#!/usr/bin/env python

import bob.fusion.base
import sklearn.preprocessing

algorithm = bob.fusion.base.algorithm.MLP(
  preprocessors=[(sklearn.preprocessing.RobustScaler(), False)])
