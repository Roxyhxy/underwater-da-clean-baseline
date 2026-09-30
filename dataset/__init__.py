from .uwd import UWD, WATER_TYPES, WATER_TYPE_TO_LABEL

__all__ = ["FLSea", "UWD", "WATER_TYPES", "WATER_TYPE_TO_LABEL"]


def __getattr__(name):
    # Keep OpenCV optional for UW-D-only tooling and tests. Existing callers
    # using ``from dataset import FLSea`` retain the same public API.
    if name == "FLSea":
        from .flsea import FLSea

        return FLSea
    raise AttributeError(name)
