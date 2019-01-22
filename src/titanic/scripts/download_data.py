"""Download data from the web."""

from titanic.tools import download_competition_data_from_kaggle


def main(path=r'data/raw'):
    download_competition_data_from_kaggle('titanic', path=path)


if __name__ == '__main__':
    main()
