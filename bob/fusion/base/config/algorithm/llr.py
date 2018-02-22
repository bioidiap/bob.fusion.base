#!/usr/bin/env python

import bob.fusion.base
from bob.fusion.base.preprocessor import LLRCalibration

algorithm = bob.fusion.base.algorithm.LLR(preprocessors=[LLRCalibration()])
