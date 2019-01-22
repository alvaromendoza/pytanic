"""Technical utility functions"""

import os
import pickle
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def download_competition_data_from_kaggle(competition, path=r'data/raw'):
    r"""Download data from a Kaggle competition.

    To use this function, sign up for a Kaggle account at ``https://www.kaggle.com``.
    Then go to the 'Account' tab of your user profile
    (``https://www.kaggle.com/<username>/account``) and select 'Create API Token'.
    This will trigger the download ofkaggle.json, a file containing your API credentials.
    Place this file in the location ``~/.kaggle/kaggle.json``
    (on Windows in the location ``C:\Users\<Windows-username>\.kaggle\kaggle.json``
    - you can check the exact location, sans drive, with ``echo %HOMEPATH%``).
    You will also need to accept competition rules at
    ``https://www.kaggle.com/c/<competition-name>/rules``.

    Parameters
    ----------
    path : str
        Where to save dataset files.

    """

    api = KaggleApi()
    api.authenticate()
    api.competition_download_cli(competition=competition, path=path)


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

    with open(file_path, 'r', encoding='utf-8') as nb:
        nbook = nbformat.read(nb, as_version=4)

#    ep = ExecutePreprocessor(kernel_name=nbook.metadata.kernelspec.name)
    ep = ExecutePreprocessor(kernel_name='python3')
    ep.preprocess(nbook, {'metadata': {'path': file_path.parent}})

    with open(file_path_temp, 'w', encoding='utf-8') as nb:
        nbformat.write(nbook, nb)

    try:
        file_path_old.unlink()
    except FileNotFoundError:
        pass

    file_path.rename(file_path_old)
    file_path_temp.rename(file_path)


def print_header(header, capitalize=False, nl_before=False, nl_after=False,
                 outline='=', outline_length=50):
    nlb, nla = '', ''
    if nl_before:
        nlb = '\n'
    if nl_after:
        nla = '\n'
    if capitalize:
        header = header.upper()
    print(nlb, outline*outline_length, sep='')
    print(header)
    print(outline*outline_length, nla, sep='')
