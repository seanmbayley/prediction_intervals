import warnings
import numpy as np
from common import NT_ModelResults

arr = np.array


def get_interval_bounds_idcs(nc):
    lower = (100 - nc) / 2.0
    upper = 100 - lower

    return int(lower * 2), int(upper * 2)


def intervals(result, nc):
    l, u = get_interval_bounds_idcs(nc)

    return result.percentiles[:, l], result.percentiles[:, u]


def mer(result, precision=2):
    residuals = np.abs(result.predictions - result.actuals)
    preds = result.predictions[:] + 1
    mers = residuals / preds

    return round(np.mean(mers), precision)


def w(result, nc, idcs=None):
    l, u = get_interval_bounds_idcs(nc)

    if idcs is None:
        idcs = arr(list(range(result.actuals.size)))

    return result.percentiles[idcs, u] - result.percentiles[idcs, l]


def _mw(result, nc, idc_sets=None):
    if idc_sets is None:
        res = np.mean(w(result, nc))
    else:
        ws = [w(result, nc, idcs) for idcs in idc_sets]
        res = np.mean(ws)

    return res


def mw(result, nc, precision=8, idc_sets=None):
    if isinstance(result, NT_ModelResults):
        w_ = _mw(result, nc, idc_sets)
    else:
        w_ = np.nan
        warnings.warn('invalid type passed: {}'.format(type(result)))

    return np.round(w_, precision)


def pc(result, nc, idc_sets=None):
    l, u = get_interval_bounds_idcs(nc)
    results = []

    idcs = list(range(result.actuals.size))
    if not idc_sets:
        idc_sets = [idcs]

    for idc_s in idc_sets:
        for i in idc_s:
            if result.percentiles[i, l] <= result.actuals[i] <= result.percentiles[i, u]:
                results.append(1)
            else:
                results.append(0)

    return results


def _coverage(result, nc, idc_sets=None):
    pc_ = pc(result, nc, idc_sets)

    return sum(pc_) / len(pc_)


def cov(result, nc, precision=2, idc_sets=None):
    if isinstance(result, NT_ModelResults):
        c = _coverage(result, nc, idc_sets)
    else:
        c = np.nan
        warnings.warn('invalid type passed: {}'.format(type(result)))

    return np.round(c, precision)


def c_and_mw(result, nc, p_c=2, p_w=8, N=None):
    idcs = list(range(result.actuals.size))
    if not N:
        idc_sets = [idcs]
    else:
        idc_sets = [np.random.choice(idcs, len(idcs)) for _ in range(N)]

    c = cov(result, nc, precision=p_c, idc_sets=idc_sets)
    w = mw(result, nc, precision=p_w, idc_sets=idc_sets)

    return c, w


def is_reliable(result, nc):
    nc_ = nc / 100
    c = cov(result, nc)

    return c >= nc_


def get_metric(result, nc, metric, precision=2):
    if metric == 'coverage':
        return cov(result, nc, precision)
    elif metric == 'mpiw':
        return mw(result, nc, precision)
    
