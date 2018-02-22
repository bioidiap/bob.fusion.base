#!/usr/bin/env python

import bob.fusion.base
from sklearn.preprocessing import StandardScaler
from bob.fusion.base.preprocessor import LLRCalibration

# algorithm = bob.fusion.base.algorithm.LLR(preprocessors=[StandardScaler(), LLRCalibration()])
algorithm = bob.fusion.base.algorithm.LLR(preprocessors=[LLRCalibration()])
