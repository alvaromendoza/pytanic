"""Technical utility functions"""

import os
import pickle
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path


def clean_directory(dir_path, keep_nested_dirs=False, files_to_keep=None):
    if files_to_keep is None:
        files_to_keep = list()
    for root, dirnames, filenames in os.walk(dir_path, topdown=False):
        for filename in filenames:
            if filename not in files_to_keep:
                os.unlink(os.path.join(root, filename))
        if not keep_nested_dirs:
            for dirname in dirnames:
                try:
                    os.rmdir(os.path.join(root, dirname))
                except OSError:
                    continue


def serialize(obj, file_path=None):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def deserialize(file_path=None):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def run_ipynb(file_path):
    """Execute Jupyter notebook programmatically.

    After all cells in the existing notebook have beer run, the function overwrites it.
    As a fail-safe, the existing notebook in its pre-run state is saved to
    'file_path.parent/.old/'.

    Parameters
    ----------
    file_path : str, pathlib.Path

    References
    ----------
    1. https://nbconvert.readthedocs.io/en/latest/execute_api.html
    2. Steven F. Lott, "Modern Python Cookbook", Chapter 9, "Replacing a file while preserving the
    previous version", pp. 437-440, 2016.

    """

    file_path = Path(file_path)
    file_path_temp = file_path.with_suffix('.ipynb.tmp')
    file_path.parent.joinpath('.old').mkdir(parents=False, exist_ok=True)
    file_path_old = (
                         file_path
                         .parent
                         .joinpath('.old', file_path.name)
                         .with_suffix('.ipynb.old')
                         )

    with open(file_path, 'r', encoding='utf-8') as nbf:
        nbook = nbformat.read(nbf, as_version=4)

    ep = ExecutePreprocessor(kernel_name=nbook.metadata.kernelspec.name)
    ep.preprocess(nbook, {'metadata': {'path': file_path.parent}})

    with open(file_path_temp, 'w', encoding='utf-8') as nbf:
        nbformat.write(nbook, nbf)

    try:
        file_path_old.unlink()
    except FileNotFoundError:
        pass

    file_path.rename(file_path_old)
    file_path_temp.rename(file_path)
