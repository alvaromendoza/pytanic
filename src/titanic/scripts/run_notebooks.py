"""Run Jupyter notebooks of the project."""

from titanic.tools import run_ipynb


def run_eda():
    run_ipynb(r'notebooks/01_exploratory_data_analysis_dummy.ipynb')


def run_compmod():
    run_ipynb(r'notebooks/02_compare_models_dummy.ipynb')


def run_notebooks():
    run_eda()
    run_compmod()


if __name__ == '__main__':
    run_notebooks()
