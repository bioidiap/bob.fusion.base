#!/usr/bin/env python

from __future__ import absolute_import, division

import logging
import pickle

from h5py import File as HDF5File

from .Algorithm import Algorithm

logger = logging.getLogger(__name__)


class AlgorithmBob(Algorithm):
    """A class to be used in score fusion using bob machines."""

    def _get_hdf5_file(self, model_file):
        return model_file[:-3] + "hdf5"

    def custom_save(self, model_file):
        # dump preprocessors in a pickle file because
        # we don't know how they look like
        # saves the class to create it later.
        with open(model_file, "wb") as f:
            pickle.dump(type(self), f)
            pickle.dump(self.preprocessors, f)
            # just for consistent string representation
            pickle.dump(self.str, f)

        d5 = HDF5File(self._get_hdf5_file(model_file), "w")
        try:
            self.machine.save(d5)
        finally:
            d5.close()

    def load(self, model_file):
        # load preprocessors and the class
        with open(model_file, "rb") as f:
            myclass = pickle.load(f)
            preprocessors = pickle.load(f)
            strings = pickle.load(f)

        myinstance = myclass(preprocessors=preprocessors)
        # just for consistent string representation
        myinstance.str.update(strings)

        d5 = HDF5File(self._get_hdf5_file(model_file))
        try:
            myinstance.machine.load(d5)
        finally:
            d5.close()

        return myinstance
