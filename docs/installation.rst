Installation
============

Via PyPI (Recommended)
----------------------

.. code-block:: bash

    pip install GaugePredict

From Source
-----------

For development:

.. code-block:: bash

    git clone https://github.com/caitlinturner/GaugePredict.git
    cd GaugePredict
    pip install -e ".[dev]"

Using Conda
-----------

.. code-block:: bash

    conda env create -f environment.yml
    conda activate gaugepredict-dev

GPU Support
-----------

For CUDA 12.6 GPU acceleration:

.. code-block:: bash

    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Requirements
------------

- Python ≥ 3.8
- PyTorch ≥ 1.10.0
- NumPy, Pandas, GeoPandas
- Scikit-learn, SHAP, Matplotlib
- See ``requirements.txt`` for complete list
