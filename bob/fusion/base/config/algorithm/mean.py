#!/usr/bin/env python

import bob.fusion.base
from bob.fusion.base.preprocessor import LLRCalibration

# two different approaches here, uncomment at will or create another algorithm
# algorithm = bob.fusion.base.algorithm.HarmonicMean()
algorithm = bob.fusion.base.algorithm.Weighted_Sum(preprocessors=[LLRCalibration()])
