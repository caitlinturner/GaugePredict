---
title: "GaugePredict: Basin-scale hydrological forecasting with hybrid CNN–LSTM models"
authors:
  - name: Caitlin R. R. Turner
    orcid: "0000-0001-8305-161X"
    affiliation: "1, 2"
  - name: Matthew Hiatt
    orcid: "0000-0003-3680-2542"
    affiliation: "1, 2"
  - name: Jo Martin
    orcid: "0009-0009-0179-6964"
    affiliation: "3"
affiliations:
  - name: Department of Oceanography and Coastal Sciences, College of the Coast and Environment, Louisiana State University, Baton Rouge, LA, United States
    index: 1
  - name: Coastal Studies Institute, Louisiana State University, Baton Rouge, LA, United States
    index: 2
  - name: Department of Geological Sciences, University of Colorado Boulder, Boulder, CO, United States
    index: 3
date: 2026-01-27
bibliography: paper.bib
---

# Summary

Hydrologic forecasting at days-to-weeks lead times supports decisions such as flood operations, navigation planning, and monitoring of threshold-based management actions. Many basins have extensive gauge networks, but building reproducible, basin-scale forecasts from multi-site time series is often slowed by data acquisition, alignment, missing-data handling, and the effort required to train and evaluate comparable models across multiple lead times.

GaugePredict is an open-source Python package that streamlines daily discharge and stage forecasting using a hybrid convolutional neural network–long short-term memory (CNN–LSTM) architecture that is widely used for time-series forecasting (for example, stock prices) [@lu2020]. In this architecture, a one-dimensional convolutional neural network (Conv1D) scans the input sequence to learn local patterns and compress them into feature maps [@lecun2002], and a long short-term memory (LSTM) recurrent network then models longer-range temporal dependencies using gated memory [@hochreiter1997]. This pairing is well-suited for basin-scale gauge predictors because the CNN efficiently extracts patterns from many input gauges while the LSTM propagates those learned patterns through time toward the forecast target [@lu2020].

GaugePredict retrieves and preprocesses multi-site daily time series (for example from the U.S. Geological Survey National Water Information System, NWIS), aligns predictors and targets onto a common timeline, constructs supervised learning sequences, trains models in PyTorch, and produces standardized evaluation outputs and figures [@usgsnwis; @paszke2019pytorch]. For interpretability and model simplification, GaugePredict uses SHapley Additive exPlanations (SHAP), a game-theoretic attribution method that assigns each predictor (gauge) a contribution to a model prediction. GaugePredict summarizes gauge importance from SHAP values and uses those rankings to identify informative predictors and optionally train reduced-input models that are easier to audit and faster to run [@lundberg2017shap; @turner2026gaugepredict].

# Statement of Need

Basin-scale, data-driven gauge forecasting is increasingly feasible, but practical adoption is often limited by workflow friction rather than model availability. Users typically need to (i) gather multi-gauge records from a data provider, (ii) reconcile inconsistent time coverage and missing values, (iii) build consistent training and evaluation setups across multiple lead times, and (iv) understand which gauges provide predictive value at different horizons. These steps are frequently assembled ad hoc, making it difficult to reproduce results, compare horizons fairly, or transfer a workflow to another basin or target gauge.

GaugePredict addresses these barriers by providing a transparent, end-to-end pipeline for daily gauge forecasting that is designed for rapid application and adaptation. The package emphasizes reproducibility through configuration-driven runs and saved artifacts, and interpretability through SHAP-based gauge ranking that can be used to reduce the predictor network while retaining comparable forecast behavior across lead times [@turner2026gaugepredict; @lundberg2017shap].

# Software Description

## Functionality

GaugePredict is organized around common user tasks in basin-scale forecasting:

- **Data retrieval and preprocessing:** retrieve daily values from NWIS, assemble basin-wide predictor time series, align predictors and a target gauge record on a shared daily index, apply consistent train/test splits, and handle missing data in a reproducible way [@usgsnwis].
- **Model training and evaluation:** train CNN–LSTM models for user-selected lead times using a standardized training interface, report comparable metrics, and export model artifacts (trained weights and preprocessing objects) for reuse [@paszke2019pytorch].
- **Explainability and predictor reduction:** compute SHAP values to attribute predictions to individual predictor gauges, summarize gauge importance, and optionally train reduced-input models using top-ranked gauges [@lundberg2017shap].
- **Visualization and diagnostics:** generate training diagnostics, observed versus predicted time series, and SHAP summary and geospatial context plots.

## Design and architecture

GaugePredict centers on a data module that prepares aligned predictor and target arrays and constructs sliding-window sequences for supervised learning. A training wrapper manages lead-time-specific training, metric logging, and artifact export, supporting consistent comparisons across horizons. Plotting utilities standardize performance figures and SHAP-based summaries so users can rapidly assess skill, errors, and predictor influence.

The software is organized into modules to support use and extension:

- `downloader`: retrieval of NWIS daily values and construction of basin-scale predictor catalogs.
- `routines`: shared utilities for preprocessing, alignment, splitting, and input/output.
- `predict`: high-level functions for training and inference at specified horizons.
- `plotting`: standardized evaluation and interpretability figures (performance, SHAP summaries, maps).

# Quality Control

GaugePredict includes unit tests for core utilities (data handling and model components) and provides example notebooks that demonstrate complete workflows from data retrieval through model evaluation and figure generation. The project documentation describes installation, configuration, and troubleshooting, and the repository distributes configuration templates to support reproducible runs across basins and target gauges [@turner2026gaugepredict].



# Example Usage

A minimal 7-day forecast workflow:

```python
from GaugePredict.predict import run_horizon

results = run_horizon(
forecast_horizon=7,
data_files="path/to/predictor/files",
use_csv_target=False,
target_site="07374000",
target_parameter_code="00060",
start_date="2020-01-01",
end_date="2023-12-31",
tz="UTC",
hp={"epochs": 50, "batch_size": 32, "sequence_length": 30},
device="cuda" # or "cpu"
)

print(results["metrics"])
```


# Impact and Reuse

GaugePredict supports basin-scale discharge and stage forecasting for short-to-medium range decisions, including operational planning and threshold-based monitoring. The combination of configuration-driven runs, exported artifacts, and standardized figures supports reproducible comparisons across lead times and predictor networks. SHAP-based gauge rankings provide a mechanism for identifying influential predictors and for building reduced-input models that can lower computational cost and improve interpretability for basin-scale applications [@lundberg2017shap].

GaugePredict is released under the MIT license and is available at the project repository and archived software release [@turner2026gaugepredict].

# Acknowledgements

Research reported in this publication was supported by the U.S. Department of Defense/Army Engineer Research and Development Center (ERDC) under Contract No. W912HZ2220005, the Gulf Research Program of the National Academies of Sciences, Engineering, and Medicine under award number SCON-10000883, and the National Science Foundation through Open Earthscape (Collaborative Research: Frameworks: OpenEarthscape - Transformative Cyberinfrastructure for Modeling and Simulation in the Earth-Surface Science Communities) award No. 2104102.

# References

