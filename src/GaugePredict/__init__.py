"""
GaugePredict

GaugePredict forecasts downstream gauge conditions using hybrid neural network models.

A comprehensive tool for predicting gauge conditions at USGS
monitoring stations or user specified locations (via csv) using 
deep learning models trained on historical U.S. Geological Survey (USGS) data.

Modules:
    downloader: Download and process USGS NWIS data
    predict: Neural network models and prediction utilities
    routines: Core utility and data processing functions
    plotting: Visualization tools for model outputs and analysis

"""

__version__ = "1.0.1"
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
