#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import

import pickle
from .Algorithm import Algorithm

import bob.core
logger = bob.core.log.setup("bob.fusion.base")


class AlgorithmBob(Algorithm):
  """A class to be used in score fusion using bob machines."""

  def _get_hdf5_file(self, model_file):
    return model_file[:-3] + 'hdf5'

  def custom_save(self, model_file):
    # dump preprocessors in a pickle file because
    # we don't know how they look like
    # saves the class to create it later.
    with open(model_file, 'wb') as f:
      pickle.dump(type(self), f)
      pickle.dump(self.preprocessors, f)
      # just for consistent string representation
      pickle.dump(self._kwargs, f)

    d5 = bob.io.base.HDF5File(self._get_hdf5_file(model_file), "w")
    try:
      self.machine.save(d5)
    finally:
      d5.close()

  def load(self, model_file):
    # load preprocessors and the class
    with open(model_file, "rb") as f:
      myclass = pickle.load(f)
      preprocessors = pickle.load(f)
      _kwargs = pickle.load(f)

    myinstance = myclass(preprocessors=preprocessors)
    # just for consistent string representation
    myinstance._kwargs.update(_kwargs)

    d5 = bob.io.base.HDF5File(self._get_hdf5_file(model_file))
    try:
      myinstance.machine.load(d5)
    finally:
      d5.close()

    return myinstance
