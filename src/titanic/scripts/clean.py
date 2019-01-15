import os
import click
from titanic.tools import clean_directory


def clean_generated_files():
    clean_directory(r'scratch_data', files_to_keep=[r'.gitkeep'])
    clean_directory(r'scratch_logs', files_to_keep=[r'.gitkeep'])
    clean_directory(r'scratch_models', files_to_keep=[r'.gitkeep'])


if __name__ == '__main__':
    clean_generated_files()
