"""Download data from the web."""

from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_titanic(path='../../data/raw'):
    r"""Download Titanic dataset from Kaggle.

    To use this function, sign up for a Kaggle account athttps://www.kaggle.com.
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
    api.competition_download_cli(competition='titanic', path=path)


if __name__ == '__main__':
    download_kaggle_titanic()
