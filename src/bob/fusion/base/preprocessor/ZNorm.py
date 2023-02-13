import numpy as np

from sklearn.preprocessing import StandardScaler

# to fix the sphinx docs
StandardScaler.__module__ = "sklearn.preprocessing"


class ZNorm(StandardScaler):
    """ZNorm feature scaler
    This scaler works just like :any:`sklearn.preprocessing.StandardScaler` but
    only takes the zero effort impostors into account when estimating the mean
    and standard deviation. You should not use this scaler when PAD scores are
    present.
    """

    def __init__(self, copy=True, **kwargs):
        """Initialize self. See help(type(self)) for accurate signature."""
        super(ZNorm, self).__init__(
            copy=copy, with_mean=True, with_std=True, **kwargs
        )

    def fit(self, X, y=None):
        """Estimates the mean and standard deviation of samples.
        Only positive samples are used in estimation.
        """
        # the fitting is done only on negative samples
        if y is not None:
            X = np.asarray(X)[~y, ...]
        return super(ZNorm, self).fit(X)
