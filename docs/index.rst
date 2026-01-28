.. image:: https://img.shields.io/badge/Python-3.8%2B-blue
    :target: https://www.python.org/downloads/
    :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-MIT-green
    :target: LICENSE
    :alt: License

.. image:: https://img.shields.io/badge/Status-Active-brightgreen
    :alt: Status

GaugePredict Documentation
==========================

GaugePredict ingests basin-wide USGS gauge time series, automatically aligns and cleans multi-site 
predictors, and generates one-day forecasts that can be used to fill gaps in target-site records 
(typically downstream conditions), as well as daily to multi-week forecasts of target-site conditions 
(eg. water level or discharge). Target-site observations can be pulled directly from USGS or ingested 
from user-provided CSV files. The model uses a hybrid neural network (CNN–LSTM), with optional 
Shapley additive explanations (SHAP)-based predictor selection to retain only the most informative 
gauges and keep the workflow lightweight on computers with limited computational capability. While 
notebooks and workflows demonstrate the package using water level and discharge forecasts in the 
Mississippi River Basin in the manuscripts and notebooks, the setup is flexible and can be applied 
to other basins, other USGS parameters, and user-defined target-site datasets.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   background
   api
   examples

Quick Navigation
================

.. grid:: 1 2 2 4
    :gutter: 3

    .. grid-item-card:: Installation
        :link: installation
        :link-type: doc

        Get up and running in minutes

    .. grid-item-card:: Quick Start
        :link: quickstart
        :link-type: doc

        Learn the basics with examples

    .. grid-item-card:: API Reference
        :link: api
        :link-type: doc

        Complete API documentation

    .. grid-item-card:: Examples
        :link: examples
        :link-type: doc

        Detailed tutorials and notebooks

Overview
--------

GaugePredict provides a complete workflow for hydrological forecasting:

* **Multi-site Data Ingestion**: Automatic download and preprocessing of USGS NWIS gauge data
* **Extended-range Forecasts**: 1–30 day prediction horizons for water level and discharge
* **Basin-wide Analysis**: Handle multiple gauge sites across entire watersheds
* **Explainability**: SHAP-based feature selection and model interpretation
* **GPU Acceleration**: Leverage CUDA for faster training on large datasets
* **Ready-to-use Workflows**: Complete notebooks for data preparation, training, and visualization

Key Features
============

 **Extended-range forecasts**
    Generate predictions from 1 to 30 days ahead for water discharge and gauge levels

 **Basin-wide analysis**
    Automatically process and align multi-site time series across entire hydrological basins

 **Explainability**
    Use SHAP values to understand feature importance and select optimal predictor sites

 **GPU acceleration**
    Support for CUDA-enabled GPUs for accelerated neural network training

 **Flexible models**
    Hybrid CNN-LSTM architectures optimized for hydrological time series

 **Workflow notebooks**
    Complete examples for every step: data preparation, training, and visualization

Core Components
===============

Downloader Module
    Retrieve USGS NWIS daily-values data and build gauge catalogs organized by hydrological unit code.

Predict Module
    Neural network architectures, training utilities, and inference functions for discharge forecasting.

Routines Module
    Core data processing and utility functions used across GaugePredict.

Plotting Module
    Comprehensive visualization tools for model predictions, SHAP analysis, and geospatial context.

Useful Links
============

* `GitHub Repository <https://github.com/your-repo>`_
* `Issues & Bug Reports <https://github.com/your-repo/issues>`_
* `Contributing Guide <https://github.com/your-repo/CONTRIBUTING>`_
* `License <https://github.com/your-repo/LICENSE>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
