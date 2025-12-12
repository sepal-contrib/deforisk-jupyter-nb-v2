Deforisk Jupyter Notebooks
==========================

A comprehensive toolkit for deforestation/degradation risk modeling and spatial analysis. This project provides Jupyter notebooks and Python scripts for processing geospatial data, building predictive models, and analyzing forest degradation patterns using machine learning and statistical methods.

📚 **Documentation**: https://deforisk-notebooks.readthedocs.io/en/latest/

Installation
------------

In sepal terminal:


.. code-block:: bash

    # Clone repo
    git clone https://github.com/SerafiniJose/deforisk-jupyter-nb-v2.git

    cd deforisk-jupyter-nb-v2

    pip install uv

    uv run main.py

    uv pip install gdal[numpy]==3.8.4 --no-build-isolation --no-cache-dir --force-reinstall

    source .venv/bin/activate

    python -m ipykernel install --user --name deforisk-deg --display-name "deforisk-deg"







