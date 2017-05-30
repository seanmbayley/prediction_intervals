import numpy as np
import pickle
import csv
import json
import os


def mf_from_mid(mstr):
    return mstr.split(',')[0].split('-')[-1]


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    :param out:
    :param arrays:
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)

    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]

    return out


def load_pickled_resource(fname):
    with open(fname, 'rb') as f:
        res = pickle.load(f)

    return res


def load_csv(fname, as_dict=True):
    data = []

    with open(fname) as f:
        if as_dict:
            reader = csv.DictReader(f)
        else:
            reader = csv.reader(f)
            reader.__next__()

        for line in reader:
            data.append(line)

    return data


def load_json(fname):
    with open(fname) as f:
        res = json.load(f)

    return res



