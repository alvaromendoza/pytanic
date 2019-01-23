"""Define command line interface of project Titanic."""

import click

import titanic.scripts.download_data
import titanic.scripts.run_notebooks
import titanic.scripts.make_features
import titanic.scripts.cross_validate_models
import titanic.scripts.make_submission
import titanic.scripts.clean_generated_files


@click.group(help='CLI entry point of project Titanic.')
def cli():
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True), default=r'data/raw')
def download(path):
    """Download Titanic competition data files from Kaggle."""
    titanic.scripts.download_data.main(path)


@cli.command()
def eda():
    """Run Jupyter notebook with exploratory data analysis."""
    titanic.scripts.run_notebooks.run_eda()


@cli.command()
def features():
    """Perform feature engineering on training and test datasets."""
    titanic.scripts.make_features.main()


@cli.command()
def crossval():
    """Cross-validate machine learning models."""
    titanic.scripts.cross_validate_models.main()


@cli.command()
def compmod():
    """Run Jupyter notebook with models comparison."""
    titanic.scripts.run_notebooks.run_compmod()


@cli.command()
def submission():
    """Make prediction on test set and create submission file."""
    titanic.scripts.make_submission.main()


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
    titanic.scripts.clean_generated_files.main(allfiles, data, logs, models, results)


if __name__ == '__main__':
    cli()
