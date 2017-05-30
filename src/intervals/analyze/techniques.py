import os
import util
import numpy as np
from collections import namedtuple
from common import NT_ModelResults, CONFIGS, SF_ALPHA
from intervals.analyze import pi
from intervals.analyze.stats import Stats
from util import load_pickled_resource


arr = np.array

TAG_DU = 'DU'
TAG_SB = 'SB'
TAG_NSB = 'NSB'
TAG_NSD = 'NSD'
TAG_SW = 'SW'
TAG_NSW = 'NSW'
TAG_ER = 'ER'
TAG_KU = 'KU'
TAG_NKU = 'NKU'
TAG_NKR = 'NKR'
TAG_TU = 'TU'


TAGS = arr([TAG_DU, TAG_SB, TAG_KU, TAG_NSD, TAG_NKU, TAG_SW, TAG_TU])
T_2_IDX = {k: v for v, k in enumerate(TAGS)}
T_2_C = {
    TAG_DU: '#3c4f09',
    TAG_SB: '#3c8f09',
    TAG_NSB: '#FFFFFF',
    TAG_KU: '#80FF00',
    TAG_NSD: '#FFFFFF',
    TAG_ER: '#FFFFFF',
    TAG_NKU: '#FF0000',
    TAG_NSW: '#FFFFFF',
    TAG_SW: '#960c0c',
    TAG_TU: '#660808',
}

ChoiceResults = namedtuple('ChoiceResults', ['a_i', 'a_t', 'm_i', 'm_t', 'weights'])


def get_config_name(technique):
    return os.path.basename(technique).split('--')[1]


def get_config_ptr(technique):
    toks = technique.split('/')
    te = toks[-2].split('_')
    te[-1] = CONFIGS
    toks[-2] = '_'.join(te)
    toks[-1] = toks[-1].split('--')[-1]

    return '/'.join(toks)


class Select:
    def __init__(self, sid):
        self._id = sid

    def __str__(self):
        return self._id

    def __repr__(self):
        return str(self)

    def pick(self, te_ptrs, techniques, nc):
        cf = None
        # index or indices of selected configuration
        res = self._pick(te_ptrs, techniques, nc)

        if res is not None:
            if isinstance(res, (list, np.ndarray)):
                if len(res):
                    cf = self.combine_cfs_by_mean(res, te_ptrs)
            else:
                cf = util.load_pickled_resource(get_config_ptr(te_ptrs[res]))

        return cf, res

    def _pick(self, te_ptrs, techniques, nc):
        raise NotImplementedError

    @staticmethod
    def combine_cfs_by_mean(cands, te_ptrs):
        actuals = None
        percentiles = None
        for i, idx in enumerate(cands):
            cf_ = util.load_pickled_resource(get_config_ptr(te_ptrs[idx]))
            if not i:
                actuals = cf_.actuals
                percentiles = cf_.percentiles
            else:
                percentiles = (i * percentiles + cf_.percentiles) / (i + 1)

        cf = NT_ModelResults(actuals, None, percentiles)

        return cf

    @staticmethod
    def get_c_and_p(techniques, nc, N=None):
        res = [pi.c_and_mw(te, nc, N=N) for te in techniques]
        c = [r[0] for r in res]
        w = [r[1] for r in res]

        return c, w

    @staticmethod
    def get_mpiws(techniques, nc):
        return arr([pi.mw(te, nc, precision=8) for te in techniques])

    @staticmethod
    def get_coverages(techniques, nc):
        return arr([pi.cov(te, nc, precision=2) for te in techniques])


class SelectA(Select):
    def __init__(self):
        super().__init__('a')

    def _pick(self, te_ptrs, techniques, nc):
        c = self.get_coverages(techniques, nc)
        w = self.get_mpiws(techniques, nc)
        nc_ = nc / 100

        cands = np.where(c >= nc_)[0]
        if cands.size:
            return cands[np.argmin(w[cands])]


class SelectB(Select):
    def __init__(self):
        super().__init__('b')

    def _pick(self, te_ptrs, techniques, nc):
        c = self.get_coverages(techniques, nc)

        cands = np.where(c >= c[-1])[0]
        if cands.size:
            return cands[np.argmin(c[cands])]


SELECTORS = [SelectA(), SelectB()]


def tag_cf_4_p(nc, d_cov, d_mpiw, s_cov, s_mpiw, p):
    d_rel = d_cov >= nc / 100
    if s_cov is None:
        tag = TAG_KU if not d_rel else TAG_TU
    else:
        s_rel = s_cov >= nc / 100

        if s_rel:
            if not d_rel:
                tag = TAG_DU
            elif d_mpiw > s_mpiw:
                tag = TAG_SB if p < SF_ALPHA else TAG_NSD
            elif s_mpiw == d_mpiw:
                tag = TAG_NSD
            else:
                tag = TAG_SW if p < SF_ALPHA else TAG_NSD
        else:
            tag = TAG_NKU if not d_rel else TAG_TU

    return T_2_IDX[tag], tag


def _load_cfs_4_t_4_p(t_ptrs):
    return [util.load_pickled_resource(t_ptr) for t_ptr in t_ptrs]


def pick_cf_4_t_4_p(t_ptrs, nc, selector, ac_4_t=None):
    if not ac_4_t:
        ac_4_t = _load_cfs_4_t_4_p(t_ptrs)
    s_cf = selector.pick(t_ptrs, ac_4_t, nc)

    return s_cf


def pick_cfs_4_t_4_p(t_ptrs, ncs, selector, ac_4_t=None):
    if not ac_4_t:
        ac_4_t = _load_cfs_4_t_4_p(t_ptrs)

    return [pick_cf_4_t_4_p(t_ptrs, nc, selector, ac_4_t=ac_4_t) for nc in ncs]


def choose_max_consensus(data):
    m = []

    for r in range(data.shape[0]):
        y = np.bincount(data[r])
        ii = np.nonzero(y)[0]
        c = np.argmax(ii)
        m.append(data[r, c])

    return m


def choose_min(choice_result, selectors, t_ptrs, nc_idx, nc):
    N_sel = len(selectors)
    m_idx = choice_result.m_i[nc_idx]
    t_idx = int(m_idx // N_sel)
    s_idx = m_idx % N_sel
    cf, _ = selectors[s_idx].pick(t_ptrs[t_idx], _load_cfs_4_t_4_p(t_ptrs[t_idx]), nc)

    return cf


def choose_weighted_vote(choice_result, selectors, t_ptrs, nc_idx, nc):
    N_sel = len(selectors)
    cf_scores = np.zeros(len(t_ptrs[0]))

    for i, t_cf_ptrs in enumerate(t_ptrs):
        cfs = _load_cfs_4_t_4_p(t_cf_ptrs)
        t_idx = i * N_sel
        for j, selector in enumerate(selectors):
            s_cf, s_idx = selector.pick(t_cf_ptrs, cfs, nc)
            if s_cf:
                cf_scores[s_idx] += choice_result.weights[nc_idx, t_idx + j]

    best_cf_idx = np.argmax(cf_scores)
    best_cf = load_pickled_resource(get_config_ptr(t_ptrs[0][best_cf_idx]))

    return best_cf


def choose_m_4_anc(at_ptrs, dcf_ptr, selectors, ncs, dcf=None):
    data = m_tags_4_anc(at_ptrs, dcf_ptr, selectors, ncs)
    a_i = np.argmin(data.sum(axis=0))
    a_t = data[:, a_i]
    m_i = np.argmin(data, axis=1)
    m_t = data.min(axis=1)
    weights = 1 / Stats.rank(data, axis=1, method='min')

    return ChoiceResults(a_i, a_t, m_i, m_t, weights)


def _coverage_mw_widths(cf, ncs):
    return [[f(cf, nc) for nc in ncs] for f in (pi.cov, pi.mw, pi.w)]


def m_tags_4_anc(at_ptrs, dcf_ptr, selectors, ncs, dcf=None, itag=True):
    if dcf is None:
        dcf = load_pickled_resource(dcf_ptr)
        d_c, d_mw, d_w = _coverage_mw_widths(dcf, ncs)
        del dcf
    else:
        d_c, d_mw, d_w = _coverage_mw_widths(dcf, ncs)

    data = []
    at_cfs = [_load_cfs_4_t_4_p(t_ptrs) for t_ptrs in at_ptrs]

    for i, nc in enumerate(ncs):
        t_data = []
        for j, t_cfs in enumerate(at_cfs):
            s_cfs = [selector.pick(at_ptrs[j], t_cfs, nc) for selector in selectors]

            data_ = []
            for s_cf, s_id in s_cfs:
                cov = mw = p = None
                if s_cf:
                    cov = pi.cov(s_cf, nc)
                    mw = pi.mw(s_cf, nc)
                    _, p = Stats.wilcoxon_signed_rank_test(pi.w(s_cf, nc), d_w[i])
                    if p != p:
                        p = 1.0

                i_tag, tag = tag_cf_4_p(nc, d_c[i], d_mw[i], cov, mw, p)
                if itag:
                    data_.append(i_tag)
                else:
                    data_.append(tag)
            t_data.extend(data_)
        data.append(t_data)

    del at_cfs
    return arr(data)


def m_tag_freqs_4_anc(at_ptrs, dcf_ptr, selectors, ncs, dcf=None):
    if not dcf:
        dcf = load_pickled_resource(dcf_ptr)

    tags = m_tags_4_anc(at_ptrs, dcf_ptr, selectors, ncs, dcf=dcf).T
    freqs = arr([np.bincount(m, minlength=TAGS.size) for m in tags])

    return freqs


def m_tag_probas_4_anc(at_ptrs, dcf_ptr, selectors, ncs, dcf=None):
    if not dcf:
        dcf = load_pickled_resource(dcf_ptr)

    freqs = m_tag_freqs_4_anc(at_ptrs, dcf_ptr, selectors, ncs, dcf=dcf)
    probas = freqs / freqs.sum(axis=1).reshape((freqs.shape[0], 1))

    return probas


def meta_tags_4_anc_4_p(mt_ptrs, t_ptrs, m_dcf_ptr, dc_ptr, selectors, ncs, itag=False):
    cr = choose_m_4_anc(mt_ptrs, m_dcf_ptr, selectors, ncs)
    dcf = load_pickled_resource(dc_ptr)
    d_c, d_mw, d_w = _coverage_mw_widths(dcf, ncs)

    tags = []
    for i, nc in enumerate(ncs):
        cf = choose_min(cr, selectors, t_ptrs, i, nc)
        cov = mw = p = None
        if cf:
            cov = pi.cov(cf, nc)
            mw = pi.mw(cf, nc)
            _, p = Stats.wilcoxon_signed_rank_test(pi.w(cf, nc), d_w[i])
            if p != p:
                p = 1.0

        idx, tag = tag_cf_4_p(nc, d_c[i], d_mw[i], cov, mw, p)
        if itag:
            tags.append(idx)
        else:
            tags.append(tag)

    return tags


def meta_tfreq_4_anc_4_p(mt_ptrs, t_ptrs, m_dcf_ptr, dcf_ptr, selectors, ncs):
    tags = meta_tags_4_anc_4_p(mt_ptrs, t_ptrs, m_dcf_ptr, dcf_ptr, selectors, ncs, itag=True)
    freqs = np.bincount(tags, minlength=TAGS.size)

    return freqs


def precision_recall_f1(true, pred):
    a_rel = np.where(true == 1)[0]
    a_nrel = np.where(true == 0)[0]
    p_rel = np.where(pred == 1)[0]
    p_nrel = np.where(pred == 0)[0]

    tp = np.intersect1d(p_rel, a_rel).size
    fp = np.setdiff1d(p_rel, a_rel).size
    fn = np.setdiff1d(p_nrel, a_nrel).size
    p = tp / (tp + fp) if (tp + fp) else (0 if fp else 1)
    r = tp / (tp + fn) if (tp + fn) else (0 if fn else 1)
    f1 = 2 * ((p * r) / (p + r)) if (p + r) else 0

    return p, r, f1, (tp + fp == 0) or (tp + fn == 0)


def clf_accuracy_4_at_4_anc(at_ptrs, cf_ptrs, ncs):
    truth_4_nc = []

    cfs = [load_pickled_resource(cfp) for cfp in cf_ptrs]
    for nc in ncs:
        truth_4_nc.append(arr([1 if pi.cov(cf, nc) >= nc / 100 else 0 for cf in cfs]))
    del cfs

    at_cfs = [_load_cfs_4_t_4_p(t_ptrs) for t_ptrs in at_ptrs]
    te_scores = []
    f1_scores_bs = []

    for i, nc in enumerate(ncs):
        truth = truth_4_nc[i]
        predictions = []
        for cfs in at_cfs:
            predictions.append(arr([1 if pi.cov(cf, nc) >= nc / 100 else 0 for cf in cfs]))

        # get the base scores
        te_scores.append([precision_recall_f1(truth, preds) for preds in predictions])

        # now bootstrap
        f1_scores = []
        for _ in range(100):
            s = np.random.choice(np.arange(truth.size), truth.size)
            # f1 for each technique
            f1 = []
            for preds in predictions:
                f1.append(precision_recall_f1(truth[s], preds[s])[2])
            f1_scores.append(f1)

        f1_scores_bs.append(f1_scores)

    del at_cfs
    return np.round(te_scores, 4), np.round(f1_scores_bs, 4)


def emmrwe(cf_ptrs, at_ptrs, ncs):
    cfs = [util.load_pickled_resource(x) for x in cf_ptrs]
    p_mmre = []
    for i, t_ptrs in enumerate(at_ptrs):
        t_mmre = []
        for j, t_ptr in enumerate(t_ptrs):
            t = util.load_pickled_resource(t_ptr)
            pw = arr([pi.mw(t, nc) for nc in ncs])
            aw = arr([pi.mw(cfs[j], nc) for nc in ncs])
            cf_mmre = (np.abs(pw - aw) / pw).reshape((1, len(ncs)))
            t_mmre.append(cf_mmre)
        p_mmre.append(np.concatenate(t_mmre))
    del cfs

    return arr([arr(p_mmre)[:, :, i] for i in range(len(ncs))])
