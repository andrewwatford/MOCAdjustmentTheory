"""Modular theory of meridional-overturning-circulation adjustment."""

from .fourier import butterworth_filter, forward_transform, inverse_transform
from .global_rossby import GlobalRossbyModel, EARTH_RADIUS_M, EARTH_ROTATION_S, rossby_speed

__all__ = [
    "GlobalRossbyModel",
    "butterworth_filter",
    "forward_transform",
    "inverse_transform",
    "rossby_speed",
    "EARTH_RADIUS_M",
    "EARTH_ROTATION_S",
]
__version__ = "0.1.0"
