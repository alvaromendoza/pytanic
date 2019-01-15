"""Run Jupyter notebooks of the project."""

from titanic.tools import run_ipynb


def run_notebooks():
    run_ipynb(r'notebooks/01_exploratory_data_analysis_dummy.ipynb')
    run_ipynb(r'notebooks/02_compare_models_dummy.ipynb')


if __name__ == '__main__':
    run_notebooks()
