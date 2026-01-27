# Contributing to GaugePredict

Thank you for your interest in contributing to GaugePredict! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues
- Search existing issues before creating a new one
- Provide a clear description of the problem
- Include steps to reproduce the issue
- Share your environment details (Python version, OS, package versions)

### Suggesting Enhancements
- Open an issue with the tag `enhancement`
- Clearly describe the proposed feature
- Explain the use case and benefits

### Pull Requests

1. **Fork the repository** and create a branch from `main`
2. **Follow the code style**:
   - Use 4 spaces for indentation
   - Follow PEP 8 guidelines
   - Add docstrings to all functions/classes
3. **Write clear commit messages**:
   - Use present tense ("Add feature" not "Added feature")
   - Reference issues: `fix: resolve issue #123`
4. **Update documentation** if needed
5. **Ensure your code works**:
   - Test locally before submitting
   - Verify examples still run
6. **Submit the pull request**:
   - Describe what changes you made and why
   - Link related issues

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

## Code Style

- Follow PEP 8
- Maximum line length: 100 characters
- Use type hints where appropriate
- Write descriptive variable names
- Add comments for complex logic

## Documentation

- Update README.md for user-facing changes
- Add docstrings following NumPy style:
  ```python
  def function_name(param1, param2):
      """
      Brief description.

      **Inputs** :
      
      param1 : 'type'
          Description of param1.
      param2 : 'type'
          Description of param2.

      **Outputs** :
      
      name : 'type'
          Description of return value.
      """
  ```

## Questions?

Contact us:
- Email: caitlin.r.r.turner@gmail.com
- Open an issue with the `question` label

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.
