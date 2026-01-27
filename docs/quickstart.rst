Quick Start
===========

Basic Usage
-----------

Download USGS gauge data:

.. code-block:: python

    from GaugePredict.downloader import download_usgs_data
    
    # Download discharge data for a site
    data = download_usgs_data(
        site_id="07374000",
        start_date="2020-01-01",
        end_date="2023-12-31",
        parameter="00060"  # Discharge
    )

Train a forecast model:

.. code-block:: python

    from GaugePredict.predict import run_horizon
    
    # Train for 7-day horizon
    results = run_horizon(
        target_site="07374000",
        horizon=7,
        sequence_length=30,
        epochs=50
    )

Complete Workflow
-----------------

See the example notebooks in ``examples/``:

1. **Data Preparation** (``downloader_notebook.ipynb``)
   - Download and preprocess basin-wide USGS data
   - Align time series and handle missing values

2. **Model Training** (``training_notebook.ipynb``)
   - Train CNN-LSTM models for multiple horizons
   - Evaluate performance with NSE, R², RMSE metrics

3. **Visualization** (``figure_creating_notebook.ipynb``)
   - Generate forecast performance plots
   - Create SHAP site selection maps

For detailed examples, see the :doc:`examples` page.
