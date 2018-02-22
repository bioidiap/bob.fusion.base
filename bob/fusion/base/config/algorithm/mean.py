#!/usr/bin/env python

import bob.fusion.base
from bob.fusion.base.preprocessor import LLRCalibration

# algorithm = bob.fusion.base.algorithm.HarmonicMean()
algorithm = bob.fusion.base.algorithm.Weighted_Sum()
# algorithm = bob.fusion.base.algorithm.Weighted_Sum(preprocessors=[LLRCalibration()])
