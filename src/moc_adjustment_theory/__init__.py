"""Global meridional-overturning-circulation adjustment theory."""

from .constants import EARTH_RADIUS, EARTH_ROTATION_RATE
from .forcing import FFTConvention
from .geometry import MultiBasinGeometry
from .model import GlobalAdjustmentModel
from .output import GlobalAdjustmentOutput

__all__ = [
    "GlobalAdjustmentModel",
    "GlobalAdjustmentOutput",
    "FFTConvention",
    "MultiBasinGeometry",
    "EARTH_RADIUS",
    "EARTH_ROTATION_RATE",
]
__version__ = "0.1.0"
