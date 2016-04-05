from .Normalizer import Normalizer
from .ZNorm import ZNorm
from .MinMaxNorm import MinMaxNorm

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
