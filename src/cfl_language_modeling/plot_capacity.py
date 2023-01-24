import argparse
import collections
import math
import pathlib
import sys

import attr
import matplotlib.pyplot as plt
import numpy

from lib.pytorch_tools.saver import read_logs
from utils.plot_util import add_plot_arguments, run_plot, get_markers

def read_plot_data(points):
    for x, dirnames in points:
        yield x, read_multiple_trials(dirnames)

@attr.s
class TrialSet:
    trials = attr.ib()
    missing = attr.ib()

def read_multiple_trials(dirnames):
    trials = []
    missing = []
    for dirname in dirnames:
        try:
            trial = read_trial(dirname)
        except IncompleteTrial:
            missing.append(dirname)
        else:
            trials.append(trial)
    return TrialSet(trials, missing)

@attr.s
class Trial:
    best_cross_entropy_diff = attr.ib()

class IncompleteTrial(Exception):

    def __init__(self, msg, dirname):
        super().__init__(msg)
        self.dirname = dirname

def read_trial(dirname):
    lower_bound_valid_perplexity = None
    best_valid_perplexity = None
    try:
        with read_logs(dirname) as events:
            for event in events:
                if event.type == 'start':
                    lower_bound_valid_perplexity = event.data['valid_lower_bound_perplexity']
                elif event.type == 'train':
                    best_valid_perplexity = event.data['best_validation_metrics']['perplexity']
    except FileNotFoundError:
        raise IncompleteTrial(f'no log file found in {dirname}', dirname) from None
    if lower_bound_valid_perplexity is None:
        raise IncompleteTrial(f'no lower bound validation perplexity found in {dirname}', dirname)
    if best_valid_perplexity is None:
        raise IncompleteTrial(f'no best validation perplexity found in {dirname}', dirname)
    best_cross_entropy_diff = compute_cross_entropy_diff(best_valid_perplexity, lower_bound_valid_perplexity)
    return Trial(best_cross_entropy_diff)

def compute_cross_entropy_diff(perplexity, lower_bound_perplexity):
    return math.log(perplexity) - math.log(lower_bound_perplexity)

def get_plot_metric(data, aggregate_func):
    return numpy.array([aggregate_func(t.best_cross_entropy_diff for t in s.trials) for s in data])

def get_plot_best(data):
    return get_plot_metric(data, min)

def get_plot_mean(data):
    return get_plot_metric(data, lambda x: numpy.mean(list(x)))

def get_plot_stddev(data):
    return get_plot_metric(data, lambda x: numpy.std(list(x)))

def add_run_annotations(ax, data, y_values, target_runs, color):
    for (xi, yi), yi_value in zip(data, y_values):
        num_runs = len(yi.trials)
        if num_runs < target_runs:
            ax.annotate(
                f'{num_runs}/{target_runs}',
                xy=(xi, yi_value),
                horizontalalignment='right',
                verticalalignment='bottom',
                textcoords='offset points',
                xytext=(-2, 2),
                color=color
            )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--label', action='append', default=[])
    parser.add_argument('--symbols', type=int, action='append', default=[])
    parser.add_argument('--inputs', type=pathlib.Path, nargs='+', action='append', default=[])
    parser.add_argument('--show-x-label', action='store_true', default=False)
    parser.add_argument('--show-stddev', action='store_true', default=False)
    add_plot_arguments(parser, outputs=['best', 'mean'])
    args = parser.parse_args()

    num_points = len(args.label)
    if len(args.symbols) != num_points:
        parser.error('different number of --label flags and --symbols flags')
    if len(args.inputs) != num_points:
        parser.error('different number of --label flags and --inputs flags')

    target_runs = max(map(len, args.inputs))

    inputs_by_label = collections.OrderedDict()
    for label, symbols, inputs in zip(args.label, args.symbols, args.inputs):
        if label not in inputs_by_label:
            inputs_by_label[label] = []
        inputs_by_label[label].append((symbols, inputs))
    labels_and_data = []
    for label, points in inputs_by_label.items():
        points.sort(key=lambda x: x[0])
        data = list(read_plot_data(points))
        for _, trial_set in data:
            for missing in trial_set.missing:
                print(f'missing: {missing}', file=sys.stderr)
        data = [(xi, yi) for xi, yi in data if yi.trials]
        x = numpy.array([xi for xi, yi in data])
        y = numpy.array([yi for xi, yi in data])
        labels_and_data.append((label, x, y))

    markers = list(get_markers(len(labels_and_data)))
    markersize = 16

    with run_plot(args, outputs=['best', 'mean']) as ((best_fig, best_ax), (mean_fig, mean_ax)):
        for (label, x, y), marker in zip(labels_and_data, markers):

            y_best = get_plot_best(y)
            #line, = best_ax.plot(x, y_best, marker=marker, markersize=markersize, label=label)
            line, = best_ax.plot(x, y_best, label=label)
            color = line.get_color()
            add_run_annotations(best_ax, zip(x, y), y_best, target_runs, color)

            y_mean = get_plot_mean(y)
            #mean_ax.plot(x, y_mean, color=color, marker=marker, markersize=markersize, label=label)
            mean_ax.plot(x, y_mean, color=color, label=label)
            if args.show_stddev:
                y_stddev = get_plot_stddev(y)
                mean_ax.fill_between(x, y_mean - y_stddev, y_mean + y_stddev, color=color, alpha=0.2)
            add_run_annotations(mean_ax, zip(x, y), y_mean, target_runs, color)

        xticks = sorted(set(args.symbols))
        for ax in (best_ax, mean_ax):
            ax.set_ylabel('Cross-entropy Diff.')
            if args.show_x_label:
                ax.set_xlabel('Alphabet Size $k$')
            ax.set_ylim(bottom=0)
            ax.set_xticks(xticks)

if __name__ == '__main__':
    main()
