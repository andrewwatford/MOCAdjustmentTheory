"""Modular theory of meridional-overturning-circulation adjustment."""

from .fourier import forward_transform, inverse_transform, GlobalAdjustmentModel
from .hello import hello_world

__all__ = ["forward_transform", "hello_world", "inverse_transform", "GlobalAdjustmentModel"]
__version__ = "0.1.0"
