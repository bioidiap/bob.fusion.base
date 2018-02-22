#!/usr/bin/env python

import bob.fusion.base
from sklearn.linear_model import LogisticRegression
from bob.fusion.base.preprocessor import LLRCalibration

algorithm = bob.fusion.base.algorithm.CascadeFuse()
# algorithm = bob.fusion.base.algorithm.CascadeFuse(preprocessors=[LLRCalibration()])
