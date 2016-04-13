from .Algorithm import Algorithm
from .Weighted_Sum import Weighted_Sum
from .LogisticRegression import LogisticRegression
from .MLP import MLP
from .MLPClassifier import MLPClassifier

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
