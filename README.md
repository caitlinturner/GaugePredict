## GaugePredict (Beta Version - V 1.0.0 Release coming soon!)
---

An open-source package that forecasts downstream gauge conditions using a hybrid neural network model.

- Creates extended-range forecasts of downstream gauge conditions (e.g., water level, discharge) from daily to multi-week horizons (1–30 days).
- Ingests and preprocesses basin-wide USGS gauge data by automatically downloading, aligning, and cleaning multi-site time series defined by user-selected HUC regions.
  - Synchronizes all sites to a continuous daily index and fills missing days so datasets are sequence-model ready.
- Trains horizon-specific CNN–LSTM models and uses SHAP-selected gauges to reduce inputs, enabling rapid forecasts on a standard machine.
- Supports continuity during data interruptions by using trained models to fill estimates when target-gauge observations are missing due to down gauges.
- Can be applied to forecast diversion-related flows for water management decisions.
- Comes with walkthrough notebooks that make dataset building, training, and figure generation easy to follow.


We welcome contributions! Please email cturn65@lsu.edu or open an issue/pull request to get involved!

---

## Installation
```bash
#git clone https://github.com/<your-username>/GaugePredict.git
#cd GaugePredict
#pip install -e .
```
---
## Workflow Assisted by Notebooks (located in examples)
- **1) GaugePredict Data Downloader notebook (`downloader_notebook.ipynb`)**
  - Builds basin-scale **predictor datasets** from USGS gauge records for GaugePredict.
  - Main steps:
    - Define a target gauge (what you want to predict) and analysis window.
    - Select upstream predictor gauges (e.g., by HUC regions) and download daily records.
    - Standardize to a continuous daily time index and handle missing values consistently.
    - Save outputs for modeling:
      - Cached time series files per gauge.
      - A site dictionary JSON with metadata.
    **This file must be created to run the other two notebooks as is. They can also be used as a guide for your own area**

- **2) GaugePredict Training notebook (`training_notebook.ipynb`)**
  - Trains the **CNN–LSTM** GaugePredict model for a selected target site and variable using the cached predictor dataset.
  - Main steps:
    - Configure run metadata (run name, target, date window, horizons).
    - Choose predictor-site selection mode:
      - Use all sites, or a SHAP-ranked subset if available (can be created in notebook).
    - Run training and evaluation per horizon.
    - Save outputs to a run-specific results folder:
      - Per-horizon predictions, metrics, and artifacts.
      - A compute summary JSON with hardware info, hyperparameters, runtimes, and skill scores for reporting.

- **3) GaugePredict Figure Creating notebook (`figure_creating_notebook.ipynb`)**
  - Generates figures from saved run outputs.
  - Main figure workflows:
    - **Training + test performance figure**
      - Loads saved runs for selected horizons.
      - Aligns test series and plots observed vs predicted plus training diagnostics.
    - **SHAP geoplot grid**
      - Reads SHAP artifacts from the SHAP results folder.
      - Plots a map grid of predictor sites and SHAP-selected subsets by horizon, with a target-site marker.


---

## License
This project is licensed under the MIT License

---

## Citing GaugePredict (NOT PEER REVIEWED YET)
If you use **GaugePredict** in your research, please cite the software and associated paper upon release.  

**Citation:**
coming soon

