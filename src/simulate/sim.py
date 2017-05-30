import argparse
import glob
import os
import pickle
import numpy as np
import data
import util
from datetime import datetime
from simulate.qrf import QRF
from multiprocessing import Pool


arr = np.array


class Sim:
    def __init__(self, num_cores):
        if num_cores == -1:
            num_cores = os.cpu_count()
        self.pool = Pool(processes=num_cores)

    def start(self, data_dir, techniques=None):
        results_dir = self._get_results_dir(data_dir)
        self.config_simulation(data_dir, results_dir)
        self.technique_simulation(techniques, data_dir, results_dir)

    @staticmethod
    def _get_results_dir(data_dir):
        results_dir = os.path.join(data.get_process_dir(data_dir), datetime.now().strftime('%m_%d_%Y'))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        return results_dir

    @staticmethod
    def _prep_dir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            files = glob.glob(dir_path + '/*')
            for f in files:
                if os.path.isdir(f):
                    os.rmdir(f)
                else:
                    os.remove(f)

    @staticmethod
    def _save(res, results_dir, rstr):
        with open(os.path.join(results_dir, '{}.p'.format(rstr)), 'wb') as f:
            pickle.dump(res, f)

    @classmethod
    def from_string(cls, arg_str):
        return cls(**cls.parse_args(arg_str))

    def technique_simulation(self, techniques, data_dir, results_dir):
        raise NotImplementedError

    def config_simulation(self, data_dir, results_dir):
        raise NotImplementedError

    @staticmethod
    def evaluate_technique(args):
        raise NotImplementedError

    @staticmethod
    def parse_args(args):
        raise NotImplementedError

    @staticmethod
    def get_configurations():
        raise NotImplementedError


class Standard(Sim):
    def __init__(self, split=0.66, meta_splits=None, meta_only=False, num_cores=-1):
        self.split = split
        self.meta_splits = meta_splits
        self.train_pct = int(100 * split)
        self.test_pct = 100 - self.train_pct
        self.meta_only = meta_only
        super().__init__(num_cores)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Standard_{}-{}'.format(self.train_pct, self.test_pct)

    def config_simulation(self, data_dir, results_dir):
        configurations = self.get_configurations()

        if not self.meta_only:
            rdir = os.path.join(results_dir, str(self), 'configurations')
            self._prep_dir(rdir)
            X, y = data.load_data(data_dir)
            for config in configurations:
                rstr = ','.join(['{}-{}'.format(k, v) for k, v in sorted(config.items(), key=lambda tup: tup[0])])
                res = QRF.evaluate_configuration(X, y, config, self.split)
                self._save(res, rdir, rstr)

        for meta_split in self.meta_splits:
            meta_rdir = os.path.join(results_dir, str(self),
                                     '{}_meta_configurations'.format(int(100 * meta_split)))
            self._prep_dir(meta_rdir)
            # discard original testing data
            X, y = data.load_data(data_dir, remove_last=self.split)
            for config in configurations:
                rstr = ','.join(['{}-{}'.format(k, v) for k, v in sorted(config.items(), key=lambda tup: tup[0])])
                res = QRF.evaluate_configuration(X, y, config, meta_split)
                self._save(res, meta_rdir, rstr)

    @staticmethod
    def evaluate_technique(args):
        config, X, y, te, train_test_split, rdir = args
        rstr = '{}--'.format(str(te)) + ','.join(['{}-{}'.format(k, v)
                                                 for k, v in sorted(config.items(), key=lambda tup: tup[0])])
        res = QRF.evaluate_technique(X, y, te, config, train_test_split)
        Sim._save(res, rdir, rstr)

    def technique_simulation(self, techniques, data_dir, results_dir):
        configurations = self.get_configurations()
        te_cross_cfgs = util.cartesian([list(range(len(techniques))), list(range(len(configurations)))])

        if not self.meta_only:
            rdir = os.path.join(results_dir, str(self), 'techniques')
            self._prep_dir(rdir)
            X, y = data.load_data(data_dir)
            batch_args = []

            for n, m in te_cross_cfgs:
                batch_args.append((configurations[m], arr(X), arr(y), techniques[n], self.split, rdir))

            self.pool.map(self.evaluate_technique, batch_args)

        for meta_split in self.meta_splits:
            meta_rdir = os.path.join(results_dir, str(self),
                                     '{}_meta_techniques'.format(int(100 * meta_split)))
            self._prep_dir(meta_rdir)

            X, y = data.load_data(data_dir, remove_last=self.split)
            meta_batch_args = []
            for n, m in te_cross_cfgs:
                meta_batch_args.append((configurations[m], X[:], y[:], techniques[n], meta_split, meta_rdir))

            self.pool.map(self.evaluate_technique, meta_batch_args)

    @staticmethod
    def get_configurations():
        rf_param_max_fts = [('max_features', round(x, 2)) for x in np.arange(0.05, 1.05, 0.05)]
        rf_parameters = [rf_param_max_fts]
        rf_param_settings = []

        for setting in util.cartesian(list(map(lambda x: range(len(x)), rf_parameters))):
            rf_param_settings.append({rf_parameters[n][m][0]: rf_parameters[n][m][1] for n, m in enumerate(setting)})

        return rf_param_settings

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-split', type=float, default=0.66)
        parser.add_argument('-meta_splits', type=float, nargs='+', default=[])
        parser.add_argument('-meta_only', action='store_true')
        parser.add_argument('-num_cores', default=1, type=int)
        parser.set_defaults(meta_only=False)
        args = parser.parse_args(args.split())
        return vars(args)

