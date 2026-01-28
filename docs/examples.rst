Examples
=========

The ``examples/`` directory contains complete, production-ready workflows for using GaugePredict.
These resources guide you through every step of the forecasting pipeline.

Jupyter Notebooks
==================

Interactive notebooks with detailed explanations and visualizations:

Data Downloader (``downloader_notebook.ipynb``)
--------------------------------------------------

Automatically download and prepare USGS gauge data for modeling.

Training Notebook (``training_notebook.ipynb``)
--------------------------------------------------

Train CNN-LSTM models on prepared gauge data with configurable horizons and hyperparameters.

Figure Creation (``figure_creating_notebook.ipynb``)
------------------------------------------------------

Generate publication-ready visualizations and interpretability plots from trained models.

Baton Rouge Missing Data Analysis (``br_gauge_missingdata_figures_notebook.ipynb``)
--------------------------------------------------------------------------------------

Demonstrates model capability for filling observational gaps during gauge downtime:

- Compare GaugePredict forecasts against measured USGS observations
- Use USGS calculated discharge as reference during missing periods
- Illustrate how upstream predictors enable discharge inference when target gauge is downed


Python Scripts
==============

``downloader_msr_basin.py``
    Download and preprocess discharge data for the Mississippi River Basin

``trainingcode_bonnet_carre_spillway.py``
    Complete training workflow for the Bonnet Carré Spillway site

``figure_creating.py``
    Generate visualizations from trained model outputs

``br_gauge_missingdata_figures.py``
    Generate forecast comparison figures for missing data scenarios (e.g., downed gauges)

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
