API Reference
=============

This section documents the public API of GaugePredict. 

For detailed tutorials and examples, see the :doc:`examples` section.

Core Package
------------

.. automodule:: GaugePredict
   :members:
   :undoc-members:
   :show-inheritance:

Downloader Module
-----------------

.. automodule:: GaugePredict.downloader
   :members:
   :undoc-members:
   :show-inheritance:

The downloader module provides utilities for retrieving USGS NWIS time series data
and assembling gauge catalogs organized by hydrological unit code (HUC).

Predict Module
--------------

.. automodule:: GaugePredict.predict
   :members:
   :undoc-members:
   :show-inheritance:

The predict module contains neural network architectures, training utilities, and
inference functions for discharge prediction and model interpretation via SHAP values.

Routines Module
---------------

.. automodule:: GaugePredict.routines
   :members:
   :undoc-members:
   :show-inheritance:

The routines module provides core utility functions used across GaugePredict,
including data processing, file I/O, and configuration management.

Plotting Module
---------------

.. automodule:: GaugePredict.plotting
   :members:
   :undoc-members:
   :show-inheritance:

The plotting module contains visualization utilities for model predictions,
SHAP summaries, and geospatial analysis.
