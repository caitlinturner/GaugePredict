# GaugePredict

[![PyPI version](https://badge.fury.io/py/GaugePredict.svg)](https://badge.fury.io/py/GaugePredict)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## An open-source package that forecasts downstream gauge conditions using a hybrid neural network.

**Note:** An update to address the dependecies for the U.S. Geological Survey's switch from dataRetrival.NWIS to dataRetrival.waterdata will be completed in the coming weeks. All functions are working at this time.


![HUC map overview](examples/figures/map_hucs.png)
*Example of Hydrologic Unit Code (HUC) selection for a desired basin.*

- Creates extended-range forecasts of downstream gauge conditions (e.g., water level, discharge) from daily to multi-week horizons (1–30 days tested).
- Ingests and preprocesses basin-wide U.S. Geological Survey gauge data by automatically downloading, aligning, and cleaning multi-site time series defined by user-selected HUC regions.
  - Synchronizes all sites to a continuous daily index and fills missing days so datasets are sequence-model ready.
- Trains a hybrid neural network model (CNN–LSTM) and uses SHAP-selected gauges to reduce inputs, enabling rapid forecasts on a standard machine.
- Supports continuity during data interruptions by using trained models to fill estimates when target-gauge observations are missing due to down gauges.
- Can be applied to forecast flows for water management decisions.
- Comes with walkthrough notebooks that make dataset building, training, and figure generation easy to follow.

![Baton Rouge missing data example](examples/figures/BR_missing.png)
*Example of filling in missing data for the Mississippi River at Baton Rouge using trained GaugePredict estimates (data source: USGS).*

![Bonnet Carré Spillway forecasts](examples/figures/training_test_bcs.png)
*GaugePredict training and test results for water level forecasts for the Mississippi River at the Bonnet Carré Spillway gauge site (data source: USACE).*


## Installation

### Via PyPI (Recommended)
```bash
pip install GaugePredict
```

### From Source (Development)
```bash
git clone https://github.com/caitlinturner/GaugePredict.git
cd GaugePredict
pip install -e ".[dev]"
```

### With Conda Environment
```bash
cd GaugePredict
conda env create -f environment.yml
conda activate gaugepredict-dev
```

### GPU Support (Optional)
For CUDA-enabled GPU acceleration:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.10.0
- NumPy, Pandas, GeoPandas
- Scikit-learn, SHAP, Matplotlib
- See `requirements.txt` for complete list



## Workflow Assisted by Notebooks (located in examples)
Also available as python scripts if notebooks aren't your thing (`examples/scripts`)
- **1) GaugePredict Data Downloader notebook (`downloader_notebook.ipynb` or `downloader_msr_basin.py`)**
  - Builds basin-scale **predictor datasets** from USGS gauge records for GaugePredict.
  - Main steps:
    - Define a target gauge (what you want to predict) and analysis window.
    - Select upstream predictor gauges (e.g., by HUC regions) and download daily records.
    - Standardize to a continuous daily time index and handle missing values consistently.
    - Save outputs for modeling:
      - Cached time series files per gauge.
      - A site dictionary JSON with metadata.
        
    **This file must be created to run the other two notebooks as is. They can also be used as a guide for your own area**

- **2) GaugePredict Training notebook (`training_notebook.ipynb` or `training_code_bonnet_carre_spillway.py`)**
  - Trains the **CNN–LSTM** GaugePredict model for a selected target site and variable using the cached predictor dataset.
  - Main steps:
    - Configure run metadata (run name, target, date window, horizons).
    - Choose predictor-site selection mode:
      - Use all sites, or a SHAP-ranked subset if available (can be created in notebook).
    - Run training and evaluation per horizon.
    - Save outputs to a run-specific results folder:
      - Per-horizon predictions, metrics, and artifacts.
      - A compute summary JSON with hardware info, hyperparameters, runtimes, and skill scores for reporting.

- **3) GaugePredict Figure Creating notebook (`figure_creating_notebook.ipynb` or `figure_creating.py`)**
  - Generates figures from saved run outputs.
  - Main figure workflows:
    - **Training + test performance figure**
      - Loads saved runs for selected horizons.
      - Aligns test series and plots observed vs predicted plus training diagnostics.
    - **SHAP geoplot grid**
      - Reads SHAP artifacts from the SHAP results folder.
      - Plots a map grid of predictor sites and SHAP-selected subsets by horizon, with a target-site marker.

- **4) Missing Data Analysis notebook (`br_gauge_missingdata_figures_notebook.ipynb` or `br_gauge_missingdata_figures.py`)**
  - Demonstrates forecasting during gauge outages and missing observations.
  - Main steps:
    - Compare GaugePredict forecasts against USGS observations and calculated discharge.
    - Illustrate how upstream predictor gauges enable discharge inference when target gauge is down.
    - Visualize forecast performance during observational gaps.

## Documentation

- [Full API Documentation](https://gaugepredict.readthedocs.io) (ReadTheDocs)
- [Contributing Guide](.github/CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## License
This project is licensed under the MIT License

## Citation

If you use **GaugePredict** in your research, please cite both the software and the associated paper:

### Software Citation
```bibtex
@software{turner2026gaugepredict,
  author = {Turner, Caitlin R. R. and Martin, Jo and Hiatt, Matthew},
  title = {{GaugePredict: Forecasting downstream gauge conditions using hybrid neural networks}},
  year = {2026},
  url = {https://github.com/caitlinturner/GaugePredict},
  version = {1.0.0}
}
```

### Paper Citation
 Coming soon!


## Contributing

We welcome contributions! Please:
- Open an issue to discuss proposed changes
- Submit pull requests for bug fixes or new features
- Contact: caitlin.r.r.turner@gmail.com

## Funding Acknowledgments

This work was supported by the US Department of Defense/Army Engineer Research and Development Center (ERDC) under Contract No. W912HZ2220005, the Gulf Research Program of the National Academies of Sciences, Engineering, and Medicine under award number SCON-10000883, and the NSF through Open Earthscape (Collaborative Research: Frameworks: OpenEarthscape - Transformative Cyberinfrastructure for Modeling and Simulation in the Earth-Surface Science Communities) award No. 2104102.
