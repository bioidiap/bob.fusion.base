#!/usr/bin/env python

import bob.fusion.base
from bob.fusion.base.preprocessor import LLRCalibration

# the algorithm itself does not matter, as we are not running it for calibration
# what is important is the preprocessor
algorithm = bob.fusion.base.algorithm.Weighted_Sum(preprocessors=[LLRCalibration()])
