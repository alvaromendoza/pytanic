import os
import click
from titanic.tools import clean_directory


def clean_generated_files():
    clean_directory(r'data - Copy', files_to_keep=[r'.gitkeep'])
    clean_directory(r'logs - Copy', files_to_keep=[r'.gitkeep'])
    clean_directory(r'models - Copy', files_to_keep=[r'.gitkeep'])


if __name__ == '__main__':
    clean_generated_files()
