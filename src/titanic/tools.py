"""Technical utility functions"""

import os
import pickle


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
