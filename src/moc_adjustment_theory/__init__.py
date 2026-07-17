"""Modular theory of meridional-overturning-circulation adjustment."""

from .fourier import forward_transform, inverse_transform
from .model import GlobalRossbyModel

__all__ = ["forward_transform", "inverse_transform", "GlobalRossbyModel"]
__version__ = "0.1.0"
