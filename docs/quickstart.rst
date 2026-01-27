Quick Start
===========


Installation
============

If you haven't installed GaugePredict yet, see the :doc:`installation` guide.

Basic Usage
===========

Downloading USGS Data
---------------------

Start by downloading discharge data for a USGS gauge site:

.. code-block:: python

    from GaugePredict.downloader import download_usgs_data
    
    # Download discharge data for a site
    data = download_usgs_data(
        site_id="07374000",
        start_date="2020-01-01",
        end_date="2023-12-31",
        parameter="00060"  # Discharge in cubic feet per second
    )
    
    print(data.head())  # View first few rows

Training a Forecast Model
--------------------------

Train a neural network model to forecast discharge. ``run_horizon`` expects
preprocessed predictor files (see the downloader notebook) plus a small
hyperparameter dictionary:

.. code-block:: python

    import torch.nn as nn
    from GaugePredict.predict import run_horizon
    
    hp = {
        "sequence_length": 30,
        "cutoff_date": "2023-01-01",
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "loss_function": nn.MSELoss(),
        "max_grad_norm": 1.0,
    }

    results = run_horizon(
        forecast_horizon=7,
        data_files="path/to/predictor/files",  # directory or file bundle produced by downloader
        use_csv_target=False,
        target_site="07374000",
        target_parameter_code="00060",
        start_date="2020-01-01",
        end_date="2023-12-31",
        tz="UTC",
        hp=hp,
        device="cuda"  # or "cpu"
    )
    
    print(f"Model Performance: R² = {results['metrics']['r2']}")

Working With Results
--------------------

The ``results`` dict returned by ``run_horizon`` includes:

- ``metrics``: R², NSE, and Willmott index for the test split
- ``y_pred`` / ``y_true``: arrays in original units
- ``history``: per-epoch training metrics
- ``model_path``: where the trained model is saved (if you set ``out_dir``)

Complete Workflow
=================

For comprehensive examples, explore the Jupyter notebooks in the ``examples/notebooks/`` directory:

**1. Data Preparation** (``downloader_notebook.ipynb``)
    - Download and preprocess basin-wide USGS data
    - Align time series across multiple gauge sites
    - Handle missing values and data quality issues

**2. Model Training** (``training_notebook.ipynb``)
    - Train CNN-LSTM models for multiple forecast horizons (1-30 days)
    - Evaluate performance with hydrological metrics (NSE, R², RMSE)
    - Visualize training history and loss curves

**3. Visualization & Analysis** (``figure_creating_notebook.ipynb``)
    - Generate forecast performance plots
    - Create SHAP site selection maps
    - Analyze feature importance across time

Common Tasks
============

.. grid:: 1 2 2 3
    :gutter: 2

    .. grid-item-card:: Download Data
        
        Retrieve USGS data using ``downloader`` module

    .. grid-item-card:: Train Models
        
        Build neural networks with ``predict`` module

    .. grid-item-card:: Visualize Results
        
        Create plots with ``plotting`` module

Next Steps
==========

- Read the :doc:`api` documentation for detailed function references
- Explore the :doc:`examples` for advanced workflows
- Check out the GitHub repository for source code and contributions

Need Help?
==========

- **Documentation**: See the :doc:`api` section for function details
- **Issues**: Report bugs on `GitHub Issues <https://github.com/caitlinturner/GaugePredict/issues>`_
- **Examples**: Look for Jupyter notebooks in ``examples/notebooks/``

For detailed examples, see the :doc:`examples` page.
