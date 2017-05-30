import os
import util
import numpy as np
from common import *


def get_process_dir(data_dir):
    data_dir = data_dir.rstrip('/')
    tokens = data_dir.split('/')
    ds_name = tokens[-1]

    if not os.path.exists(PROCESSED_DDIR):
        os.mkdir(PROCESSED_DDIR)

    pdir = os.path.join(PROCESSED_DDIR, ds_name)

    return pdir


def _read_load_cfg(data_dir, data):
    load_cfg = util.load_json(os.path.join(data_dir, 'load_cfg.json'))

    load_cfg[LOAD_CATEGORICAL_OPTS] = {}

    for cf in load_cfg[LOAD_CATEGORICAL]:
        load_cfg[LOAD_CATEGORICAL_OPTS][cf] = set()

    for line in data:
        for cf in load_cfg[LOAD_CATEGORICAL]:
            load_cfg[LOAD_CATEGORICAL_OPTS][cf].add(line[cf])

    return load_cfg


def _combine_csvs(data_dir):
    data = []
    csvs = [os.path.join(data_dir, x) for x in os.listdir(data_dir)
            if x.split('.')[-1] == 'csv']
    sub_dirs = [x for x in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, x))]

    for sd in sub_dirs:
        csvs.extend([os.path.join(data_dir, sd, x) for x in os.listdir(os.path.join(data_dir, sd))
                     if x.split('.')[-1] == 'csv'])

    for csv_ in csvs:
        data.extend(util.load_csv(csv_))

    return data


def load_data(data_dir, remove_last=None):
    data = _combine_csvs(data_dir)
    load_cfg = _read_load_cfg(data_dir, data)
    X = []
    y = []

    for n, line in enumerate(data):
        datum = []
        for k in sorted(line.keys()):
            if k not in load_cfg[LOAD_TO_EXCLUDE]:
                if k in load_cfg[LOAD_CATEGORICAL]:
                    for ft_ in load_cfg[LOAD_CATEGORICAL_OPTS][k]:
                        if line[k] == ft_:
                            datum.append(1)
                        else:
                            datum.append(0)
                elif k == load_cfg[LOAD_CLASS]:
                    y.append(float(line[k]))
                else:
                    if line[k] != '?':
                        datum.append(float(line[k]))
                    else:
                        datum.append(np.nan)

        X.append(datum)

    X = np.asarray(X)
    y = np.asarray(y)

    if remove_last:
        n = int(X.shape[0] * remove_last)
        X, y = X[:n], y[:n]

    return X, y


def dirs_to_process(data_dir, recursive):
    dirs = []
    if recursive:
        for sd in [os.path.join(data_dir, sd) for sd in os.listdir(data_dir)]:
            if os.path.isdir(sd):
                dirs.append(sd)
    else:
        dirs.append(data_dir)

    return [x for x in dirs if os.path.basename(x) not in PROJECTS_TO_EXCLUDE]