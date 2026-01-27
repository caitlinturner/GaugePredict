"""
GaugePredict - Forecasts downstream gauge conditions using hybrid neural network models
"""

__version__ = "1.0.0"
__author__ = "Caitlin R. R. Turner"
__email__ = "cturn65@lsu.edu"

# Import main modules for easy access
from . import downloader
from . import predict
from . import routines
from . import plotting

__all__ = [
    "downloader",
    "predict",
    "routines",
    "plotting",
    "__version__",
]
