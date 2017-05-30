import warnings
import numpy as np
from intervals.analyze.stats import Stats


def _prep(sample):
    sample_ = sample[:]

    if isinstance(sample, np.ndarray):
        sample_ = sample_.tolist()

    return sample_


def tag_counts(sample, tags):
    sample_ = _prep(sample)
    counts = []

    for t in tags:
        counts.append(sample_.count(t))

    return np.array(counts)


def tag_probas(sample, tags):
    N = len(sample)
    counts = tag_counts(sample, tags)

    return counts / N


def tag_errors(mu, sigma, N, CI):
    l, u = Stats.interval(mu, sigma, CI, N)

    if l != l:
        l = 0
    if u != u:
        u = 0

    l_e = min(mu, mu - l)
    u_e = min(1 - mu, u - mu)

    return l_e, u_e
