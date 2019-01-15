"""Run exploratory data analysis Jupyter notebook."""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path


def run_notebook(notebook_path):
    """Generic funcion to execute Jupyter notebooks programmatically.

    After all cells in the existing notebook have beer run, the function overwrites it.
    As a fail-safe, the existing notebook in its pre-run state is saved in 'notebook_path/old/'.

    Parameters
    ----------
    notebook_path : str, pathlib.Path

    References
    ----------
    1. https://nbconvert.readthedocs.io/en/latest/execute_api.html
    2. Steven F. Lott, "Modern Python Cookbook", Chapter 9, "Replacing a file while preserving the
    previous version", pp. 437-440, 2016.

    """

    notebook_path = Path(notebook_path)
    notebook_path_temp = notebook_path.with_suffix('.ipynb.tmp')
    notebook_path.parent.joinpath('.old').mkdir(parents=False, exist_ok=True)
    notebook_path_old = (
                         notebook_path
                         .parent
                         .joinpath('.old', notebook_path.name)
                         .with_suffix('.ipynb.old')
                         )

    with open(notebook_path, 'r', encoding='utf-8') as nbf:
        nbook = nbformat.read(nbf, as_version=4)

    ep = ExecutePreprocessor(kernel_name=nbook.metadata.kernelspec.name)
    ep.preprocess(nbook, {'metadata': {'path': notebook_path.parent}})

    with open(notebook_path_temp, 'w', encoding='utf-8') as nbf:
        nbformat.write(nbook, nbf)

    try:
        notebook_path_old.unlink()
    except FileNotFoundError:
        pass

    notebook_path.rename(notebook_path_old)
    notebook_path_temp.rename(notebook_path)


if __name__ == '__main__':
    run_notebook('notebooks/01_exploratory_data_analysis_dummy.ipynb')
