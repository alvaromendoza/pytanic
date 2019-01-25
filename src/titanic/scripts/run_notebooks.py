"""Run Jupyter notebooks of the project."""

from titanic.tools import run_ipynb


def run_eda():
    run_ipynb(r'notebooks/01_exploratory_data_analysis.ipynb')


def run_compmod():
    run_ipynb(r'notebooks/02_compare_models.ipynb')


def main():
    run_eda()
    run_compmod()


if __name__ == '__main__':
    main()
