import numpy as np
from common import RF_RANDOM_STATE, NT_ModelResults
from sklearn.ensemble import RandomForestRegressor


arr = np.array


class QRF:
    @staticmethod
    def make_rf(**kwargs):
        return RandomForestRegressor(**kwargs, random_state=RF_RANDOM_STATE, min_samples_leaf=1, n_estimators=1000)

    @staticmethod
    def make_intervals(clf, X_test):
        responses = [[] for _ in range(X_test.shape[0])]

        for est in clf.estimators_:
            for n, leaf_idx in enumerate(est.apply(X_test)):
                response = est.tree_.value[leaf_idx][0]
                if np.setdiff1d(response, np.unique(response)).size:
                    exit('node is not pure')

                responses[n].extend(response)

        # noinspection PyTypeChecker
        intervals = arr([np.percentile(response, np.arange(0, 100.5, 0.5)) for response in responses])

        return intervals

    @staticmethod
    def evaluate(clf, X_train, X_test, y_train, y_test):
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        intervals = QRF.make_intervals(clf, X_test)

        return NT_ModelResults(y_test, preds, intervals)

    @staticmethod
    def train_test_split(X, split_pct):
        split = int(X.shape[0] * split_pct)
        return np.arange(split), np.arange(split, X.shape[0])

    @staticmethod
    def evaluate_technique(X, y, te, config, split):
        train, test = QRF.train_test_split(X, split)
        clf = QRF.make_rf(**config)

        preds = []
        actuals = []
        intervals = []

        for train_, test_ in te.split(train):
            r = QRF.evaluate(clf, X[train_], X[test_], y[train_], y[test_])
            preds.extend(r.predictions)
            actuals.extend(r.actuals)
            intervals.extend(r.percentiles)

        res = NT_ModelResults(arr(actuals), arr(preds), arr(intervals))

        return res

    @staticmethod
    def evaluate_configuration(X, y, config, split):
        train, test = QRF.train_test_split(X, split)
        clf = QRF.make_rf(**config)

        return QRF.evaluate(clf, X[train], X[test], y[train], y[test])
