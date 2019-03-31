#!/usr/bin/env python
"""
Convert dos linefeeds (crlf) to unix (lf).
    This module is responsible for making pickle files compatible
with Python 3.6. Orginally sourced the method from StackOverFlow,
re-optimized a few times for more efficient repeated operations.
"""

import pickle
from pickle import UnpicklingError

# Private Functions
def _to_unix(path):
    """ converts a file to unix endings """
    original = path
    destination = original.replace(".pkl", "_unix.pkl")

    content = ''
    outsize = 0
    with open(original, 'rb') as inpath:
        content = inpath.read()
    with open(destination, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))
    return destination

# Functions
def pickle_load(path):
    """ Load pickle paths in Python 3 """
    try:
        data = pickle.load(open(path, "rb"))
        return data
    except UnpicklingError:
        unix_path = path.replace(".pkl", "_unix.pkl")
        try:
            data = pickle.load(open(unix_path, "rb"))
            return data
        except FileNotFoundError:
            pass
        except UnpicklingError:
            pass
        path = _to_unix(path)
        data = pickle.load(open(path, "rb"))
        return data
