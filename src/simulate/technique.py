import argparse
import numpy as np
from sklearn import model_selection


class PETechnique:
    def __init__(self, tid):
        self._id = tid

    def split(self, X, y=None):
        raise NotImplementedError

    def __str__(self):
        return self._id

    def __repr__(self):
        return str(self)

    @classmethod
    def from_string(cls, args):
        return cls(**cls.parse_args(args))

    @staticmethod
    def parse_args(args):
        raise NotImplementedError


class HxKCV(PETechnique):
    def __init__(self, h_reps=10, k_folds=10):
        self.h = h_reps
        self.k = k_folds
        super().__init__('10x10fold')

    def split(self, X, y=None):
        folds = []
        partition = int(X.shape[0] // self.k)

        for _ in range(self.h):
            idcs = list(range(X.shape[0]))
            np.random.seed(_)
            np.random.shuffle(idcs)
            for i in range(self.k):
                start = i * partition
                stop = start + partition
                train = np.asarray(np.r_[idcs[:start], idcs[stop:]], dtype=np.int32)
                test = np.asarray(idcs[start:stop], dtype=np.int32)
                folds.append((train, test))

        return folds

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-h_reps', type=int, default=10)
        parser.add_argument('-k_folds', type=int, default=10)
        return vars(parser.parse_args(args.split()))


class LeaveOneOut(PETechnique):
    def __init__(self):
        super().__init__('LOO')

    def split(self, X, y=None):
        folds = []
        loo = model_selection.LeaveOneOut()

        for train, test in loo.split(X):
            folds.append((train, test))

        return folds

    @staticmethod
    def parse_args(args):
        return {}


class TSHVCV(PETechnique):
    def __init__(self, gamma=0.25, delta=0.5, folds=10):
        self.gamma = gamma
        self.delta = delta
        self.k_folds = folds
        super().__init__('TSHVCV')

    def split(self, X, y=None):
        """
        h = n * gamma
        v = (n - n^delta - 2h - 1) / 2

        """
        folds = []

        h = int(X.shape[0] * self.gamma)
        v = int((X.shape[0] - X.shape[0] ** self.delta - 2 * h - 1) / 2)

        step_ = int((X.shape[0] - v - v) / self.k_folds)

        for i in range(v, X.shape[0] - v, step_):
            v_block = np.arange(i - v, i + v + 1)
            h_block = np.arange(max(0, i - v - h), min(X.shape[0], i + v + h + 1))

            train_left = np.arange(0, h_block[0])
            train_right = np.arange(h_block[-1] - 1, X.shape[0])
            train = np.r_[train_left, train_right]
            folds.append((train, v_block))

        return folds

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-gamma', type=float, default=0.25)
        parser.add_argument('-delta', type=float, default=0.5)
        parser.add_argument('-folds', type=float, default=10)

        return vars(parser.parse_args(args.split()))


class TSCV(PETechnique):
    def __init__(self, n=10):
        self.n = n
        super().__init__('TSCV')

    def split(self, X, y=None):
        tscv = model_selection.TimeSeriesSplit(n_splits=self.n)

        return tscv.split(X)

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-n', type=int, default=10)

        return vars(parser.parse_args(args.split()))


class Split(PETechnique):
    def __init__(self, p=0.75):
        self.p = p
        train, test = int(self.p * 100), 100 - int(self.p * 100)
        super().__init__('{}-{}'.format(train, test))

    def split(self, X, y=None):
        split = int(X.shape[0] * self.p)

        return [(np.arange(0, split), np.arange(split, X.shape[0]))]

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', type=float, default=0.75)

        return vars(parser.parse_args(args.split()))


class BootstrapOOS(PETechnique):
    def __init__(self, n=100):
        self.n = n
        super().__init__('{}_OOS_Bootstrap'.format(self.n))

    def split(self, X, y=None):
        folds = []
        indices = list(range(X.shape[0]))

        for _ in range(self.n):
            train = np.random.choice(indices, size=X.shape[0])
            test = list(set(indices) - set(train.tolist()))
            folds.append((train, test))

        return folds

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-n', type=int, default=100)

        return vars(parser.parse_args(args.split()))


def get_all_default_tes():
    te = ['BootstrapOOS', 'Split -p 0.25', 'Split -p 0.50', 'Split -p 0.75', 'TSCV', 'TSHVCV',
          'LeaveOneOut', 'HxKCV']

    return ['technique.{}'.format(t) for t in te]

