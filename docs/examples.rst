Examples
========

The ``examples/`` directory contains complete workflow demonstrations:

Jupyter Notebooks
-----------------

1. **Data Downloader** (``downloader_notebook.ipynb``)
   
   - Download USGS gauge data for predictor sites
   - Define target gauge and analysis window
   - Preprocess and cache time series data

2. **Training Notebook** (``training_notebook.ipynb``)
   
   - Configure CNN-LSTM model hyperparameters
   - Train models for multiple forecast horizons
   - Evaluate performance metrics
   - Generate predictions

3. **Figure Creation** (``figure_creating_notebook.ipynb``)
   
   - Load trained model results
   - Create performance visualizations
   - Generate SHAP site selection maps

Python Scripts
--------------
Example scripts for code functionality if notebooks arent your thing
- **downloader_msr_basin.py**: Example data downloading script for Mississippi River Basin
- **trainingcode_bonnet_carre_spillway.py**: Example training script for Bonnet Carré Spillway site
- **figure_creating.py**: Examplefigure creation code 

Data Files
----------

- ``bcs_wl.csv``: Example water level data
- ``cached_data_discharge/``: Cached discharge data (Not included here, can be created in ``downloader_notebook.ipynb``)
- ``results/``: Model outputs and predictions
- ``shapefiles/``: HUC zone and basin shapefiles

For detailed usage, open the notebooks in Jupyter or JupyterLab.
