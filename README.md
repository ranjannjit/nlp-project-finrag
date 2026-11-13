# NLP Project FinRAG

This project runs on **Python 3.11** managed via Conda and Poetry.

## Quick Start Setup (One-time only)
These steps do not need to be run every time:
1. Install Miniconda.
2. Create the Python 3.11 environment (`conda create -y --name finrag python=3.11`).
3. Install Poetry (`conda run -n finrag pip install poetry`).
4. Install project dependencies (`conda run -n finrag poetry install`).

## Running the Project Sequence
To execute applications and scripts using the active virtual environment, run them in the following order:

### 1. Run the Streamlit Application
```bash
conda run --no-capture-output -n finrag streamlit run app.py
```

### 2. Run Python Scripts or Notebooks
Ensure you prefix your commands with `conda run -n finrag` to execute them inside the Python 3.11 context:
```bash
conda run -n finrag python <your_script>.py
```
