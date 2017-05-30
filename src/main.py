import argparse
import importlib
import os
import data
import numpy as np
import re
from simulate.technique import get_all_default_tes


def _parse_bool(s):
    return s in ['True', 'true']


def _parse_float(s):
    try:
        val = float(s)
    except ValueError:
        val = np.nan

    return val


def ap_t_intervals_op(cla):
    """
    Usage: -op "core.<PIOperation> <params [...]>"
    :type cla: str
    """
    pi_op = 'intervals.' + cla.split()[0]
    module = importlib.import_module('.'.join(pi_op.split('.')[:-1]))
    pi_op_class = getattr(module, pi_op.split('.')[-1])
    return pi_op_class.from_string(' '.join(cla.split()[1:]))


def ap_t_intervals(cla):
    """
    Usage: -p "core.<Analyzer> <params [...]>"
    :type cla: str
    """
    pi_analyzer = 'intervals.' + cla.split()[0]
    module = importlib.import_module('.'.join(pi_analyzer.split('.')[:-1]))
    pi_analyzer_class = getattr(module, pi_analyzer.split('.')[-1])
    return pi_analyzer_class.from_string(' '.join(cla.split()[1:]))


def ap_t_techniques(cla):
    """
    Usage: -t "technique.<T1> <params [...]>" "technique.<T2> <params [...]>" ...
    :type cla: str
    """
    if cla == 'all':
        return cla
    te = 'simulate.' + cla.split()[0]
    module = importlib.import_module('.'.join(te.split('.')[:-1]))
    te_class = getattr(module, te.split('.')[-1])
    return te_class.from_string(' '.join(cla.split()[1:]))


def ap_t_sim(cla):
    """
    Usage: -s "sim.<S> <params [...]>"
    :type cla: str
    """
    s = 'simulate.' + cla.split()[0]
    module = importlib.import_module('.'.join(s.split('.')[:-1]))
    sim_class = getattr(module, s.split('.')[-1])
    return sim_class.from_string(' '.join(cla.split()[1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = ap.add_subparsers(dest='action')

    ap_sim = subparsers.add_parser('simulate')
    ap_sim.add_argument('-d', '--sim_data',
                        default='../data/raw/ant')
    default_techniques = ['technique.Split -p 0.25',
                          'technique.Split -p 0.5',
                          'technique.Split -p 0.75']
    ap_sim.add_argument('-t', '--techniques',
                        type=ap_t_techniques,
                        nargs='+',
                        default=list(map(ap_t_techniques, default_techniques)),
                        help=ap_t_techniques.__doc__)
    ap_sim.add_argument('-s', '--sim',
                        type=ap_t_sim,
                        default='sim.Standard -meta_splits 0.25 0.50 0.75 -num_cores -1',
                        help=ap_t_sim.__doc__,)

    ap_predict = subparsers.add_parser('intervals')
    ap_predict.add_argument('-m', '--interval_manager',
                            type=ap_t_intervals,
                            default='core.PIManager -d ../data/processed/ant -ncs 90 95 99',
                            help=ap_t_intervals.__doc__)
    ap_predict.add_argument('-op', '--interval_operation',
                            type=ap_t_intervals_op,
                            default='core.Intervals -mtry 1.0 -out ant_default_intervals')

    # ap_sim.set_defaults()
    args = ap.parse_args()

    if args.action == 'simulate':
        if len(args.techniques) == 1:
            if args.techniques[0] == 'all':
                args.techniques = [ap_t_techniques(t) for t in get_all_default_tes()]

        args.sim.start(args.sim_data, args.techniques)
    elif args.action == 'intervals':
        args.interval_operation.evaluate(args.interval_manager)
    else:
        exit('invalid action, choose from: (\'simulate\', \'intervals\')')

if __name__ == '__main__':
    main()