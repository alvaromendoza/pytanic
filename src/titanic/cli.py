import click
from titanic.scripts.clean import clean_generated_files
from titanic.scripts.download_data import download_kaggle_titanic
from titanic.scripts.make_features import make_features
from titanic.scripts.run_notebooks import run_eda
from titanic.scripts.run_notebooks import run_compmod
from titanic.scripts.cross_validate_models import cross_validate_models
from titanic.scripts.make_submission import make_submission


@click.group(help='CLI entry point of Titanic project.')
def cli():
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True), default=r'data/raw')
def download(path):
    """Download Titanic competition data files from Kaggle."""
    download_kaggle_titanic(path)


@cli.command()
def eda():
    """Run Jupyter notebook with exploratory data analysis."""
    run_eda()


@cli.command()
def features():
    """Perform feature engineering on training and test datasets."""
    make_features()


@cli.command()
def crossval():
    """Cross-validate machine learning models."""
    cross_validate_models()


@cli.command()
def compmod():
    """Run Jupyter notebook with models comparison."""
    run_compmod()


@cli.command()
def submission():
    """Make prediction on test set and create submission file."""
    make_submission()


@cli.command()
@click.option('--allfiles', '-a', is_flag=True, default=False,
              help='Delete all files.')
@click.option('--data', '-d', is_flag=True, default=False,
              help='Delete files only in data directory.')
@click.option('--logs', '-l', is_flag=True, default=False,
              help='Delete files only in logs directory.')
@click.option('--models', '-m', is_flag=True, default=False,
              help='Delete files only in models directory.')
@click.option('--results', '-r', is_flag=True, default=False,
              help='Delete files only in results directory.')
def clean(allfiles, data, logs, models, results):
    """Delete automatically generated files."""
    clean_generated_files(allfiles, data, logs, models, results)


if __name__ == '__main__':
    cli()
