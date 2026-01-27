Installation
=============

Choose your preferred installation method below.

Via PyPI (Recommended)
======================

The easiest way to get started is using pip:

.. code-block:: bash

    pip install GaugePredict

This installs the latest stable release with all core dependencies.

From Source
===========

For development or to use the latest features:

.. code-block:: bash

    git clone https://github.com/caitlinturner/GaugePredict.git
    cd GaugePredict
    pip install -e ".[dev]"

The ``-e`` flag installs in editable mode, allowing you to modify the source code
and see changes immediately.

Using Conda
===========

For a conda-based environment:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate gaugepredict-dev

This creates an isolated environment with all required dependencies.

GPU Support (Optional)
======================

GaugePredict supports GPU acceleration via PyTorch and CUDA. Install the CUDA version
of PyTorch for your system:

**For CUDA 12.6:**

.. code-block:: bash

    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

**For other CUDA versions:**

Visit `PyTorch's installation guide <https://pytorch.org/get-started/locally/>`_ 
to find the correct command for your system.

GPU acceleration is optional—the package works fine with CPU-only PyTorch.

Requirements
============

**Core Requirements:**

- Python ≥ 3.8
- PyTorch ≥ 1.10.0
- NumPy
- Pandas
- GeoPandas
- Scikit-learn
- SHAP
- Matplotlib

See ``requirements.txt`` in the repository for the complete list of dependencies
and their versions.

Verification
============

Verify the installation was successful:

.. code-block:: python

    import GaugePredict
    print(GaugePredict.__version__)

This should print the version number (e.g., ``1.0.1``) without errors.

Troubleshooting
===============

**Import Error: "No module named 'GaugePredict'"**
    Ensure the package is installed in your active Python environment.
    Run ``pip list | grep GaugePredict`` to verify.

**PyTorch Installation Issues**
    Refer to the official `PyTorch installation guide <https://pytorch.org/>`_
    for your specific operating system and hardware.

**GPU Not Detected**
    Run ``torch.cuda.is_available()`` to check if CUDA is properly installed.
    For support, see the PyTorch CUDA compatibility guide.

If you encounter other issues, please report them on the 
`GitHub Issues page <https://github.com/caitlinturner/GaugePredict/issues>`_.
