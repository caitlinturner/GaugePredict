# Changelog

All notable changes to GaugePredict will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-01-27

### Changed
- Updated author information with all three co-authors
- Updated citation metadata with correct ORCID IDs and affiliations
- Updated email addresses for package maintainers

### Fixed
- Corrected paper.bib journal citation for JGR submission

## [1.0.0] - 2026-01-27

### Added
- Initial release of GaugePredict
- CNN-LSTM neural network for gauge forecasting
- SHAP-based feature selection for predictor sites
- Support for 1-30 day forecast horizons
- USGS data downloader with automatic preprocessing
- Example notebooks for complete workflow
- PyPI package distribution
- Comprehensive documentation and citation files

### Features
- Multi-site time series ingestion and alignment
- Horizon-specific model training
- GPU acceleration support (CUDA 12.6)
- Automated data cleaning and gap handling
- Geographic visualization of predictor sites
- Model performance metrics (NSE, R², RMSE)

[1.0.1]: https://github.com/caitlinturner/GaugePredict/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/caitlinturner/GaugePredict/releases/tag/v1.0.0
