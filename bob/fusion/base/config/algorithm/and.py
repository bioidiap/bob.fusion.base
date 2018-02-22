#!/usr/bin/env python

import bob.fusion.base
from bob.fusion.base.preprocessor import LLRCalibration

# this should be called on calibrated scores only
# calibration ensures that different systems with different classifiers and scores are compatible
algorithm = bob.fusion.base.algorithm.CascadeFuse(preprocessors=[LLRCalibration()])
