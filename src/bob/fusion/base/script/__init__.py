from .fuse import routine_fusion  # noqa: F401

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
