from .Algorithm import Algorithm
from .Weighted_Sum import Weighted_Sum
from .LLR import LLR
from .MLP import MLP

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
