import os
import argparse
import util
import numpy as np
from common import SF_ALPHA
from intervals.visualize import Visualize
from intervals.analyze import configs
from intervals.analyze import techniques
from intervals.analyze.stats import Stats

arr = np.array


class PIManager:
    def __init__(self, data_dir, ncs):
        self.data_dir = data_dir
        self.ncs = ncs
        self.rdir = os.path.join('/'.join(data_dir.split('/')[:-2]), 'results')

        if not os.path.isdir(self.rdir):
            os.makedirs(self.rdir)

    @classmethod
    def from_string(cls, arg_str):
        return cls(**cls.parse_args(arg_str))

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-data_dir')
        parser.add_argument('-ncs', nargs='+', type=int, default=(90, 95, 99))
        args = parser.parse_args(args.split())
        return vars(args)

    def get_cf_ptrs(self, loc):
        configurations = []

        for cfg in os.listdir(os.path.join(self.data_dir, loc)):
            if cfg[-2:] == '.p':
                configurations.append(os.path.join(self.data_dir, loc, cfg))

        return sorted(configurations)

    def load_configurations(self, loc):
        return [util.load_pickled_resource(cfp) for cfp in self.get_cf_ptrs(loc)]

    def get_default_ptr(self, loc):
        return self.get_cf_ptrs(loc)[-1]

    def load_default_cfg(self, loc):
        return util.load_pickled_resource(self.get_default_ptr(loc))

    def get_at_ptrs(self, loc):
        techniques = []
        for te in os.listdir(os.path.join(self.data_dir, loc)):
            if te[-2:] == '.p':
                techniques.append(os.path.join(self.data_dir, loc, te))

        techniques = [sorted([te for te in techniques if te.split('--')[0] == x])
                      for x in sorted(list(set([y.split('--')[0] for y in techniques])))]

        return techniques

    def get_t_idx(self, t_id, loc):
        for i, t in enumerate(self.get_at_ptrs(loc)):
            if os.path.basename(t[0]).split('--')[0] == t_id:
                return i

    def format_rstr(self, rstr):
        return os.path.join(self.rdir, rstr)


class PIOperation:
    def __init__(self):
        self.viz = Visualize

    @classmethod
    def from_string(cls, arg_str):
        return cls(**cls.parse_args(arg_str))

    @staticmethod
    def parse_args(args):
        raise NotImplementedError

    def evaluate(self, manager):
        raise NotImplementedError


class Intervals(PIOperation):
    def __init__(self, mtry, cf_loc, out):
        self.mtry = mtry
        self.loc = cf_loc
        self.out = out
        super().__init__()

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-mtry', type=str, default='1.0')
        parser.add_argument('-cf_loc', type=str, default='configurations')
        parser.add_argument('-out', type=str, default='intervals')
        args = parser.parse_args(args.split())
        return vars(args)

    def evaluate(self, manager):
        cf_ptrs = manager.get_cf_ptrs(self.loc)
        mtrys = [os.path.basename(cfp).split('-')[-1].split('.p')[0] for cfp in cf_ptrs]

        if self.mtry in mtrys:
            rstr = manager.format_rstr(self.out)
            cf = util.load_pickled_resource(cf_ptrs[mtrys.index(self.mtry)])
            colors = ['g', 'b', 'y']
            data = []
            observed_y = []

            for i, nc in enumerate(manager.ncs):
                if i < len(colors):
                    c = colors[i]
                else:
                    # default to white
                    c = 'w'

                low, high = configs.intervals(cf, nc)
                observed_y.extend(high)
                data.append(((low, high), c, nc))

            self.viz.plot_intervals(cf.actuals, cf.predictions, data, rstr,
                                    title='\n',
                                    xlabel='Data-points',
                                    ylabel='Response',
                                    ylim=(-1, max(np.max(observed_y), np.max(cf.actuals) + np.std(cf.actuals))))

        else:
            exit('can\'t find configuration {}, are you sure it is there?'.format(self.mtry))


class Coverage(PIOperation):
    def __init__(self, cf_loc, out):
        self.loc = cf_loc
        self.out = out
        super().__init__()

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-cf_loc', type=str, default='configurations')
        parser.add_argument('-out', type=str, default='coverage')
        args = parser.parse_args(args.split())
        return vars(args)

    def evaluate(self, manager):
        rstr = manager.format_rstr(self.out)
        cfs = manager.load_configurations(self.loc)
        covs = configs.coverage_acf_anc_4_p(cfs, manager.ncs)
        pvalues = configs.coverage_test_acf_anc_4_p(cfs, manager.ncs)

        xlabels = []
        for i, nc in enumerate(manager.ncs):
            pct_rel = np.where(covs[:, i] >= nc / 100)[0].size / covs.shape[0]
            sig = pvalues[i] <= SF_ALPHA
            xlabels.append('{}{} ({}%)'.format(nc, '*' if sig else '', int(pct_rel * 100)))

        self.viz.box(covs, rstr,
                     xlabel='Nominal Confidence',
                     ylabel='Coverage',
                     xlabels=xlabels,
                     yticks=np.arange(0, 1.2, 0.2),
                     ylim=(-0.1, 1.1))


class Width(PIOperation):
    def __init__(self, cf_loc, out):
        self.loc = cf_loc
        self.out = out
        super().__init__()

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-cf_loc', type=str, default='configurations')
        parser.add_argument('-out', type=str, default='width')
        args = parser.parse_args(args.split())
        return vars(args)

    def evaluate(self, manager):
        rstr = manager.format_rstr(self.out)
        cfs = manager.load_configurations(self.loc)
        widths = configs.width_acf_anc_4_p(cfs, manager.ncs)
        pvalues = configs.width_test_acf_anc_4_p(cfs, manager.ncs)

        xlabels = []
        for i, nc in enumerate(manager.ncs):
            sig = pvalues[i] <= SF_ALPHA
            xlabels.append('{}{}'.format(manager.ncs[i], '*' if sig else ''))

        self.viz.box(widths, rstr,
                     title='\n',
                     xlabel='Nominal Confidence',
                     ylabel='Width',
                     xlabels=xlabels)


class F1_EMMRE(PIOperation):
    def __init__(self, cf_loc, te_loc, out):
        self.cf_loc = cf_loc
        self.te_loc = te_loc
        self.out = out
        super().__init__()

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-cf_loc', type=str, default='configurations')
        parser.add_argument('-te_loc', type=str, default='techniques')
        parser.add_argument('-out', type=str, default='f1_emmre')
        args = parser.parse_args(args.split())
        return vars(args)

    def evaluate(self, manager):
        rstr = manager.format_rstr(self.out)
        cf_ptrs = manager.get_cf_ptrs(self.cf_loc)
        at_ptrs = manager.get_at_ptrs(self.te_loc)

        # F1
        scores, bs_f1_scores = techniques.clf_accuracy_4_at_4_anc(at_ptrs, cf_ptrs, manager.ncs)
        f1 = scores[:, :, 2].T.reshape(-1, len(manager.ncs))
        pvalues = [Stats.friedmans(bs_f1_scores[n, :, :].T)[1] for n in range(bs_f1_scores.shape[0])]
        xlabels = ['{}{}'.format(nc, '*' if pvalue <= SF_ALPHA else '') for nc, pvalue in zip(manager.ncs, pvalues)]

        self.viz.box(f1, rstr + '_f1',
                     title='\n',
                     xlabel='Nominal Confidence',
                     ylabel='F1',
                     xlabels=xlabels)

        # EMMRWE
        emmrwe = techniques.emmrwe(cf_ptrs, at_ptrs, manager.ncs)
        pvalues = [Stats.friedmans(emmrwe[i])[1] for i in range(len(manager.ncs))]
        emmrwe = emmrwe.reshape(3, -1).T
        xlabels = ['{}{}'.format(nc, '*' if pvalue <= SF_ALPHA else '') for nc, pvalue in zip(manager.ncs, pvalues)]

        self.viz.box(emmrwe,
                     rstr + '_emmre',
                     title='\n',
                     xlabel='Nominal Confidence',
                     ylabel='EMMRWE',
                     xlabels=xlabels)


class Tuning(PIOperation):
    def __init__(self, cf_loc, te_loc, out):
        self.cf_loc = cf_loc
        self.te_loc = te_loc
        self.out = out
        super().__init__()

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-cf_loc', type=str, default='configurations')
        parser.add_argument('-te_loc', type=str, default='techniques')
        parser.add_argument('-out', type=str, default='tuning')
        args = parser.parse_args(args.split())
        return vars(args)

    def evaluate(self, manager):
        rstr = manager.format_rstr(self.out)
        dcf_ptr = manager.get_default_ptr(self.cf_loc)
        at_ptrs = manager.get_at_ptrs(self.te_loc)
        at_ids = [os.path.basename(te_[0].split('--')[0]) for te_ in at_ptrs]

        selectors = [techniques.SelectA()]
        s_ids = [str(s) for s in selectors]
        tags = techniques.m_tags_4_anc(at_ptrs, dcf_ptr, selectors, manager.ncs, itag=False).T
        tags_ = np.split(tags, len(at_ptrs), axis=0)
        data = []

        for n, te_tags in enumerate(tags_):
            t = arr([np.repeat([None], len(s_ids))]).T
            t[0][0] = at_ids[n]
            data.extend(np.concatenate((t, te_tags), axis=1))

        header = ['technique'] + ['NC{}'.format(nc) for nc in manager.ncs]
        self.viz.table(header, data, rstr)


class MetaTuning(PIOperation):
    def __init__(self, cf_loc, te_loc, meta_splits, out):
        self.cf_loc = cf_loc
        self.te_loc = te_loc
        self.meta_splits = meta_splits
        self.out = out
        super().__init__()

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-cf_loc', type=str, default='configurations')
        parser.add_argument('-te_loc', type=str, default='techniques')
        parser.add_argument('-meta_splits', type=str, nargs='+', default=('25',))
        parser.add_argument('-out', type=str, default='meta_tuning')
        args = parser.parse_args(args.split())
        return vars(args)

    def evaluate(self, manager):
        meta_cf = '{}_meta_' + self.cf_loc
        meta_te = '{}_meta_' + self.te_loc

        rstr = manager.format_rstr(self.out)
        dcf_ptr = manager.get_default_ptr(self.cf_loc)
        at_ptrs = manager.get_at_ptrs(self.te_loc)

        meta_tags = []
        meta_ids = []

        if self.meta_splits is not None:
            for p in self.meta_splits:
                meta_dcf_ptr = manager.get_default_ptr(meta_cf.format(p))
                meta_t_ptrs = manager.get_at_ptrs(meta_te.format(p))

                t_meta_min = techniques.meta_tags_4_anc_4_p(meta_t_ptrs, at_ptrs,
                                                            meta_dcf_ptr, dcf_ptr,
                                                            techniques.SELECTORS,
                                                            manager.ncs,
                                                            itag=False)
                meta_ids.append(['meta-{}/{}'.format(p, 100 - int(p))])
                meta_tags.append(t_meta_min)

        meta_tags = arr(meta_tags)
        meta_ids = arr(meta_ids)
        data = np.concatenate((meta_ids, meta_tags), axis=1)
        header = ['technique'] + ['NC{}'.format(nc) for nc in manager.ncs]
        self.viz.table(header, data, rstr)