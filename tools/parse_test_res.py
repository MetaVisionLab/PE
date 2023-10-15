"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Or run

$ python tools/parse_test_res.py output/my_experiment --last-five[ --seed seed]

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp

Or run

$ python tools/parse_test_res.py output/my_experiment --multi-exp --last-five[ --seed seed]
"""
import re
import numpy as np
import os.path as osp
import argparse

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_dir(directory, end_signal, regex_acc, regex_err, regex_best, args):
    print('Parsing {}'.format(directory))
    subdirs = listdir_nohidden(directory, sort=True)

    if args.last_five:
        assert args.seed in subdirs
        subdirs = [args.seed]

    valid_fpaths = []
    valid_accs = []
    valid_errs = []
    valid_best = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, 'log.txt')
        assert check_isfile(fpath)
        good_to_go = False
        tmp_fpaths = []
        tmp_accs = []
        tmp_errs = []
        tmp_bests = []
        with open(fpath, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                match_acc = regex_acc.search(line)
                if match_acc:
                    acc = float(match_acc.group(1))
                    tmp_accs.append(acc)
                    tmp_fpaths.append(fpath)

                match_err = regex_err.search(line)
                if match_err:
                    err = float(match_err.group(1))
                    tmp_errs.append(err)

                match_best = regex_best.search(line)
                if match_best:
                    best = float(match_best.group(1))
                    tmp_bests.append(best)
        if args.last_five:
            valid_fpaths = tmp_fpaths[-6:-1]
            valid_accs = tmp_accs[-6:-1]
            valid_errs = tmp_errs[-6:-1]
            valid_best= [tmp_bests[-1]] * 5
        else:
            valid_fpaths.append(tmp_fpaths[-1])
            valid_accs.append(tmp_accs[-1])
            valid_errs.append(tmp_errs[-1])
            valid_best.append(tmp_bests[-1])

    for fpath, acc, err, best in zip(valid_fpaths, valid_accs, valid_errs, valid_best):
        print('file: {}. acc: {:.2f}%. err: {:.2f}%. best: {:.2f}%'.format(fpath, acc, err, best))

    acc_mean = np.mean(valid_accs)
    acc_std = compute_ci95(valid_accs) if args.ci95 else np.std(valid_accs)

    err_mean = np.mean(valid_errs)
    err_std = compute_ci95(valid_errs) if args.ci95 else np.std(valid_errs)

    best_mean = np.mean(valid_best)
    best_std = compute_ci95(valid_best) if args.ci95 else np.std(valid_best)

    print('===')
    print('outcome of directory: {}'.format(directory))
    if args.res_format in ['acc', 'acc_and_err', 'acc_and_best']:
        print('* acc: {:.2f}±{:.2f}'.format(acc_mean, acc_std))
    if args.res_format in ['err', 'acc_and_err']:
        print('* err: {:.2f}±{:.2f}'.format(err_mean, err_std))
    if args.res_format in ['best', 'acc_and_best']:
        print('* best: {:.2f}±{:.2f}'.format(best_mean, best_std))
    print('===')

    return acc_mean, err_mean, best_mean


def parse_test_res():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help='Path to directory')
    parser.add_argument(
        '--ci95',
        action='store_true',
        help=r'Compute 95\% confidence interval'
    )
    parser.add_argument(
        '--test-log', action='store_true', help='Process test log'
    )
    parser.add_argument(
        '--multi-exp', action='store_true', help='Multiple experiments'
    )
    parser.add_argument('--last-five', action='store_true', help='Return mean and std of last five epoch.')
    parser.add_argument('--seed', type=str, default='1')
    parser.add_argument(
        '--res-format',
        type=str,
        default='acc_and_best',
        choices=['acc', 'err', 'best', 'acc_and_err', 'acc_and_best']
    )
    args = parser.parse_args()
    end_signal = 'Finished training'
    if args.test_log:
        end_signal = '=> result'
    regex_acc = re.compile(r'\* accuracy: ([\.\deE+-]+)%')
    regex_err = re.compile(r'\* error: ([\.\deE+-]+)%')
    regex_best = re.compile(r'\* Best Acc: ([\.\deE+-]+)%')

    if args.multi_exp:
        accs, errs, bests = [], [], []
        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)
            acc, err, best = parse_dir(
                directory, end_signal, regex_acc, regex_err, regex_best, args
            )
            accs.append(acc)
            errs.append(err)
            bests.append(best)
        acc_mean = np.mean(accs)
        err_mean = np.mean(errs)
        best_mean = np.mean(bests)
        print('overall average')
        if args.res_format in ['acc', 'acc_and_err', 'acc_and_best']:
            print('* acc: {:.2f}%'.format(acc_mean))
        if args.res_format in ['err', 'acc_and_err']:
            print('* err: {:.2f}%'.format(err_mean))
        if args.res_format in ['best', 'acc_and_best']:
            print('* best: {:.2f}%'.format(best_mean))
    else:
        parse_dir(args.directory, end_signal, regex_acc, regex_err, args)


if __name__ == '__main__':
    parse_test_res()
