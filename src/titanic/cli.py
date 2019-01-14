import click
from titanic.scripts.clean import clean_generated_files


@click.group()
def cli():
    pass


@cli.command()
def clean():
    clean_generated_files()


if __name__ == '__main__':
    cli()
