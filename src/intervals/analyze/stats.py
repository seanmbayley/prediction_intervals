import math
import numpy as np
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from scipy.stats import friedmanchisquare
from scipy.stats import norm
from scipy.stats import normaltest
from scipy.stats import rankdata
from scipy.stats import wilcoxon
from common import SF_ALPHA
from intervals.analyze import pi


arr = np.array


class Stats:
    @staticmethod
    def cochrans(X):
        if not isinstance(X, (np.ndarray,)):
            X = arr(X)

        k = X.shape[1]
        N = X.sum()

        x = ((X.sum(axis=0) - (N / k))**2).sum()
        y = (X.sum(axis=1) * (k - X.sum(axis=1))).sum()

        q = (k * (k - 1)) * (x / y)
        p = 1 - chi2.cdf(q, k - 1)

        return q, p

    @staticmethod
    def pc_contingency_table(models, C):
        pcs = [pi.pc(model, C) for model in models]
        c_tbl = []

        for pc in pcs:
            c_tbl.append([sum(pc), len(pc) - sum(pc)])

        return c_tbl

    @staticmethod
    def chi2_homogeneity(c_tbl):
        return chi2_contingency(c_tbl)

    @staticmethod
    def coverage_chi2_homogeneity(models, C, alpha=SF_ALPHA):
        chi2, p, dof, exp = Stats.chi2_homogeneity(Stats.pc_contingency_table(models, C))

        return p < alpha, p

    @staticmethod
    def friedmans(measurements):
        return friedmanchisquare(*measurements)

    @staticmethod
    def width_friedmans(models, C, alpha=SF_ALPHA):
        widths = np.array([pi.w(model, C) for model in models])
        fc2, p = Stats.friedmans(widths)

        return p < alpha, p

    @staticmethod
    def rank_configs_by_width(models, C):
        mpiws = [pi.mw(model, C) for model in models]
        return rankdata(mpiws, 'max')

    @staticmethod
    def interval(mu, sigma, c, N=1):
        return norm.interval(c, loc=mu, scale=sigma / math.sqrt(N))

    @staticmethod
    def is_normal(a, axis=0):
        return normaltest(a, axis=axis)

    @staticmethod
    def wilcoxon_signed_rank_test(x, y):
        return wilcoxon(x, y)

    @staticmethod
    def rank(a, axis=0, method='average'):
        if a.shape[0] == a.size:
            ranks = rankdata(a, method)
        else:
            if axis == 0:
                ranks = [rankdata(a[:, c], method) for c in range(a.shape[1])]
            else:
                ranks = [rankdata(a[r, :], method) for r in range(a.shape[0])]

        return arr(ranks)

