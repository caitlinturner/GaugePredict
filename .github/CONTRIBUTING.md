# Contributing to GaugePredict

(Language adapted from https://passah2o.github.io/dorado/misc/contributing.html)

We welcome contributions to the GaugePredict package in the form of pull requests and issues made in the source repository.

## Issues

If you are having any problems using GaugePredict we suggest opening an issue. When you open an issue, please provide a clear description of what your scenario is, and what error message you are receiving. If possible, please include a minimal working example of some code that breaks, and the error output you receive.

If there is some functionality you would like to see added to GaugePredict you can also open an issue up to discuss that. This can be code you plan to write and contribute, or it can be something you would like to have available but are not comfortable coding yourself. Either way we are happy to help!

## Pull Requests

If you have a feature that you would like to propose be integrated into GaugePredict, then you should open a pull request. To create a pull request, we recommend first forking the repository, and then creating a separate branch to develop your feature (for reference see the [GitHub flow guide](https://guides.github.com/introduction/flow/) and this [Git branching guide](https://learngitbranching.js.org/)). Then you can commit and develop your feature in your branch. We ask that new features be accompanied by additional unit tests, to ensure that they operate as expected. For unit testing, we use `pytest`. When you are satisfied with the code you have developed, you can open a pull request to the "main" branch of the project repository. Please write a concise and descriptive title for the pull request, and provide a clear description of what the feature does and why you are proposing its addition to the project. Before developing any new code, you are more than welcome to open an issue first to discuss your proposed addition.

## Code Style

To ensure consistency within the codebase, we follow some standard Python conventions for our formatting. As much as possible we try to stick to the [PEP-8](https://peps.python.org/pep-0008/) standard for our code. For docstrings, we try to follow the [PEP-257](https://peps.python.org/pep-0257/) standard.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/GaugePredict.git
cd GaugePredict

# Create development environment
conda env create -f environment.yml
conda activate gaugepredict-dev

# Install in editable mode
pip install -e ".[dev]"
```

## Questions?

Contact us:
- Email: caitlin.r.r.turner@gmail.com
- Open an issue with the `question` label

## Code of Conduct

Be respectful, inclusive, and professional in all interactions and do not be afraid to ask questions!
