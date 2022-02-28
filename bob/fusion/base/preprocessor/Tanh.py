import numpy as np

from sklearn.preprocessing import StandardScaler

# to fix the sphinx docs
StandardScaler.__module__ = "sklearn.preprocessing"


class Tanh(StandardScaler):
    """A tanh feature scaler:

    .. math::

        0.5 \\left( \\tanh\\left( 0.01 \\cdot \\frac{X - \\mu}{\\sigma}\\right) + 1 \\right)

    This scaler is both efficient and is robust to outliers.

    The original implementation in ``Hampel, Frank R., et al. "Robust
    statistics: the approach based on influence functions." (1986).`` uses an
    influence function but this is not used here.
    """

    def __init__(self, copy=True, **kwargs):
        """Initialize self. See help(type(self)) for accurate signature."""
        super(Tanh, self).__init__(
            copy=copy, with_mean=True, with_std=True, **kwargs
        )

    def fit(self, X, y=None):
        """Estimates the mean and standard deviation of samples.
        Only positive samples are used in estimation.
        """
        # the fitting is done only on positive samples
        if y is not None:
            X = np.asarray(X)[y, ...]
        return super(Tanh, self).fit(X)

    def transform(self, X, copy=None):
        """Perform scaling."""
        X = super(Tanh, self).transform(X, copy=copy)
        X *= 0.01
        X = np.tanh(X, out=X)
        X += 1
        X *= 0.5
        return X

    def inverse_transform(self, X, copy=None):
        """Perform inverse scaling."""
        copy = copy if copy is not None else self.copy
        if copy:
            X = X.copy()
        X *= 2
        X -= 1
        X = np.arctanh(X, out=X)
        X *= 100
        return super(Tanh, self).inverse_transform(X, copy=copy)
