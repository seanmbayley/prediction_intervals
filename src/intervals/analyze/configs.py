import numpy as np
from common import SF_ALPHA
from intervals.analyze import pi
from intervals.analyze.stats import Stats
from intervals.analyze.techniques import T_2_IDX, TAG_KU, TAG_NSD
from util import load_pickled_resource

arr = np.array

TAG_AU = 'AU'
TAG_DU = 'DU'
TAG_SB = 'SB'
TAG_NSB = 'NSB'
TAG_E = 'E'
TAG_2_IDX = {
    TAG_DU: 0,
    TAG_SB: 1,
    TAG_NSB: 3,
    TAG_AU: 2,
    TAG_E: 4
}

IDX_2_TAG = {v: k for k, v in TAG_2_IDX.items()}
TAG_2_C = {
    TAG_DU: '#3c4f09',
    TAG_SB: '#3c8f09',
    TAG_NSB: '#FFFFFF',
    TAG_AU: '#3cef09',
    TAG_E: '#FFFFFF'
}
TAGS = arr([x[1] for x in sorted([(v, k) for k, v in TAG_2_IDX.items()])])

TAG_2_TE_IDX = {
    TAG_DU: T_2_IDX[TAG_DU],
    TAG_SB: T_2_IDX[TAG_SB],
    TAG_NSB: T_2_IDX[TAG_NSD],
    TAG_AU: T_2_IDX[TAG_KU],
    TAG_E: T_2_IDX[TAG_NSD]
}

arr = arr


def _get_configs(acs):
    if isinstance(acs[0], str):
        return [load_pickled_resource(c_ptr) for c_ptr in acs]
    else:
        return acs[:]


def tag_cf_4_p(nc, d_cov, d_mpiw, cf_cov, cf_mpiw, p):
    nc_ = nc / 100

    if cf_cov is None:
        cf_cov = 0

    if cf_cov < nc_:
        tag = TAG_AU if d_cov < nc_ else TAG_E
    elif d_cov >= nc_:
        if d_mpiw <= cf_mpiw:
            tag = TAG_E
        else:
            if p < SF_ALPHA:
                tag = TAG_SB
            else:
                tag = TAG_NSB
    else:
        tag = TAG_DU

    return tag


def get_tag_ac_4_p(acs, nc, cache=None):
    """

    :param acs: all configurations.
                acs can either be a list of strings or list of NT_ModelResults.

                *********************************************************
                    acs should already be sorted by MTRY, i.e,
                    the default configuration (MTRY=1.0) MUST BE LAST
                *********************************************************
    :param nc: the nominal confidence
    :param cache: this function gets called a lot, so we cache all the NT_ModeResults.
    :return: scenario tag
    """
    nc_ = nc / 100

    if isinstance(acs[0], str):
        c_ptrs_ = acs[:]
        if cache is None:
            cache = {}

        if not cache:
            default_ptr = c_ptrs_.pop(-1)
            default_cfg = load_pickled_resource(default_ptr)
            cfgs = [load_pickled_resource(c_ptr) for c_ptr in c_ptrs_]
            for c_ptr, cfg in zip(acs, cfgs + [default_cfg]):
                cache[c_ptr] = cfg
        else:
            default_ptr = c_ptrs_.pop(-1)
            default_cfg = cache[default_ptr]
            cfgs = [cache[c_ptr] for c_ptr in c_ptrs_]
    else:
        default_cfg = acs[-1]
        cfgs = acs[:-1]

    d_cov = pi.cov(default_cfg, nc)
    d_mpiw = pi.mw(default_cfg, nc, precision=8)
    coverages = arr([pi.cov(cfg, nc) for cfg in cfgs])
    candidates = np.where(coverages >= nc_)[0]
    cf_cov = cf_mpiw = p = None

    if candidates.size:
        mpiws = arr([pi.mw(cfgs[i], nc, precision=8) for i in candidates])
        selected = mpiws.argmin()
        z, p = Stats.wilcoxon_signed_rank_test(pi.w(default_cfg, nc),
                                               pi.w(cfgs[candidates[selected]], nc))
        cf_cov = coverages[candidates[selected]]
        cf_mpiw = mpiws[selected]

    return tag_cf_4_p(nc, d_cov, d_mpiw, cf_cov, cf_mpiw, p)


def tags_acf_anc_4_p(ac_ptrs, ncs, cache=None):
    return arr([get_tag_ac_4_p(ac_ptrs, nc, cache) for nc in ncs])


def tags_acf_anc_4_ap(ac_4_ap_ptrs, ncs, to_exclude=None):
    idcs = list(range(len(ac_4_ap_ptrs[0])))
    if to_exclude:
        idcs = list(set(idcs) - set(to_exclude))

    tags_by_nc = []

    for ac_ptrs in ac_4_ap_ptrs:
        cache = {}
        tags_by_nc.append(tags_acf_anc_4_p(arr(ac_ptrs)[idcs].tolist(), ncs, cache))

    return arr(tags_by_nc).T


def width_acf_anc_4_p(acs, ncs):
    data = []

    acs_ = _get_configs(acs)

    for c in ncs:
        data.append([pi.mw(cf, c, precision=6) for cf in acs_])

    return np.transpose(data)


def width_test_acf_anc_4_p(acs, ncs):
    res = []
    acs_ = _get_configs(acs)

    for c in ncs:
        a_w = arr([pi.w(cf, c) for cf in acs_])
        _, p = Stats.friedmans(a_w)
        res.append(p)

    return res


def coverage_test_acf_anc_4_p(acs, ncs):
    res = []
    acs_ = _get_configs(acs)

    for nc in ncs:
        a_pc = arr([pi.pc(cf, nc) for cf in acs_])
        q, p = Stats.cochrans(a_pc.T)
        res.append(p)

    return res


def coverage_acf_anc_4_p(acs, ncs):
    res = []
    acs_ = _get_configs(acs)
    for c in ncs:
        res.append([pi.cov(cf, c, precision=2) for cf in acs_])

    return np.transpose(res)


def ideal_tags_4_p(acs, ncs, itag=False):
    tags = []
    cache = {}
    acs_ = _get_configs(acs)

    for i, nc in enumerate(ncs):
        tag = get_tag_ac_4_p(acs_, nc, cache)
        if itag:
            tags.append(TAG_2_TE_IDX[tag])
        else:
            tags.append(tag)

    return tags


def intervals(cf, nc):
    if isinstance(cf, str):
        cf = load_pickled_resource(cf)

    lower, upper = pi.intervals(cf, nc)

    return lower, upper
