"""Technical utility functions"""

import pickle


def serialize(obj, file_path=None):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def deserialize(file_path=None):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
