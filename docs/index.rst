GaugePredict Documentation
==========================

Welcome to GaugePredict's documentation!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Overview
--------

GaugePredict is an open-source Python package for forecasting downstream gauge conditions
using hybrid CNN-LSTM neural networks. It provides tools for:

* Multi-site USGS gauge data ingestion and preprocessing
* Horizon-specific (1-30 day) forecast model training
* SHAP-based feature selection for predictor sites
* Automated data cleaning and gap handling
* Model evaluation and visualization

Key Features
------------

- **Extended-range forecasts**: 1–30 day horizons for water level and discharge
- **Basin-wide analysis**: Automatic download and alignment of multi-site time series
- **SHAP selection**: Reduce inputs using explainability-driven site selection
- **GPU acceleration**: Support for CUDA-enabled training
- **Workflow notebooks**: Complete examples for data preparation, training, and visualization

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
