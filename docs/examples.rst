Examples
=========

The ``examples/`` directory contains complete, production-ready workflows for using GaugePredict.
These resources guide you through every step of the forecasting pipeline.

Jupyter Notebooks
==================

Interactive notebooks with detailed explanations and visualizations:

Data Downloader (``downloader_notebook.ipynb``)
--------------------------------------------------

Learn how to download and prepare USGS gauge data:

- Download time series from multiple USGS gauge sites
- Define target gauges and analysis time windows
- Preprocess data and handle missing values
- Cache processed data for faster access in training
- Organize data by hydrological unit code (HUC)

Training Notebook (``training_notebook.ipynb``)
--------------------------------------------------

Build and train neural network forecasting models:

- Configure CNN-LSTM model architectures
- Set hyperparameters for different forecast horizons (1-30 days)
- Train models on historical discharge data
- Evaluate performance using hydrological metrics (NSE, R², RMSE)
- Generate predictions for validation periods
- Save trained models for deployment

Figure Creation (``figure_creating_notebook.ipynb``)
------------------------------------------------------

Visualize and interpret model results:

- Load trained model outputs and predictions
- Create performance comparison plots
- Generate forecast accuracy visualizations
- Produce SHAP feature importance analysis
- Create site selection maps for predictor gauges
- Export figures for reports and publications


Python Scripts
==============

Standalone Python scripts demonstrating specific workflows:

``downloader_msr_basin.py``
    Download and preprocess discharge data for the Mississippi River Basin

``trainingcode_bonnet_carre_spillway.py``
    Complete training workflow for the Bonnet Carré Spillway site

``figure_creating.py``
    Generate visualizations from trained model outputs

Data & Resources
================

The ``examples/`` directory includes:

**Data Files:**

- ``bcs_wl.csv``: Sample water level dataset for Bonnet Carré Spillway
- ``data/``: Additional input data files

**Cached Data:**

- ``cached_data_discharge/``: Pre-processed discharge data
  
  (Can be generated using ``downloader_notebook.ipynb``)

**Results:**

- ``results/``: Model outputs, predictions, and metrics
- ``results/*/H01/`` and ``results/*/H03/``: Results for different forecast horizons

**Geospatial Data:**

- ``shapefiles/HUC_Zones/``: Hydrological unit code (HUC) boundaries
- ``shapefiles/MSRB/``: Mississippi River Basin boundary
- ``shapefiles/US_STATES/``: US state boundaries

Getting Started
===============

**Option 1: Interactive Notebooks (Recommended)**

1. Open Jupyter Lab in the repository:

   .. code-block:: bash

       jupyter lab

2. Navigate to ``examples/notebooks/``
3. Start with ``downloader_notebook.ipynb``
4. Follow the notebooks in order

**Option 2: Run Scripts**

1. Navigate to the examples directory:

   .. code-block:: bash

       cd examples/scripts


Tips & Best Practices
=====================

 **Data Preparation**

- Always download data before training
- Cache data to avoid repeated API calls
- Check for data gaps and quality issues

 **Model Training**

- Start with shorter prediction horizons (1-3 days)
- Use GPU acceleration for faster training
- Monitor validation metrics during training

 **Visualization**

- Create multiple horizon forecasts for comparison
- Use SHAP plots to understand model decisions
- Validate results against real-world observations

Need Help?
==========

- See :doc:`api` for function documentation
- Check GitHub Issues for common problems
- Contribute improvements via pull requests
