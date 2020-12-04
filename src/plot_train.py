import argparse
import itertools
import math
import pathlib

import attr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy
import tikzplotlib

from nsrnn.pytorch_tools.saver import read_logs

def read_data(dirname, metric):
    trials = [read_data_for_trial(d, metric) for i, d in list_dirs_in_order(dirname)]
    if metric == 'cross-entropy':
        for trial in trials:
            take_log(trial, 'perplexity', 'cross-entropy')
    if metric == 'perplexity-diff':
        for trial in trials:
            subtract_lower_bounds(trial, 'perplexity', 'perplexity-diff')
    if metric == 'cross-entropy-diff':
        for trial in trials:
            subtract_lower_bounds(trial, 'perplexity', 'cross-entropy-diff', math.log)
    return average_trials(trials, metric)

def list_dirs_in_order(dirname):
    pairs = ((int(d.name), d) for d in dirname.iterdir())
    return sorted(pairs, key=lambda x: x[0])

@attr.s
class Trial:
    lower_bound_dev_perplexity = attr.ib()
    dev_metrics = attr.ib()
    test_lengths = attr.ib()
    test_metrics = attr.ib()
    lower_bound_test_perplexities = attr.ib()
    overall_test_metrics = attr.ib()
    overall_lower_bound_test_perplexity = attr.ib()

METRICS = ('perplexity', 'accuracy')

def read_data_for_trial(dirname, metric):
    lower_bound_dev_perplexity = None
    dev_metrics = { k : [] for k in METRICS }
    test_lengths = []
    test_metrics = { k : [] for k in METRICS }
    lower_bound_test_perplexities = []
    overall_test_metrics = None
    overall_lower_bound_test_perplexity = None
    with read_logs(dirname) as events:
        for event in events:
            if event.type == 'start':
                lower_bound_dev_perplexity = event.data['valid_lower_bound_perplexity']
            elif event.type == 'validate':
                for k in METRICS:
                    dev_metrics[k].append(event.data[k])
            elif event.type == 'test_length':
                test_lengths.append(event.data['length'])
                for k in METRICS:
                    test_metrics[k].append(event.data[k])
                lower_bound_test_perplexities.append(event.data['lower_bound_perplexity'])
            elif event.type == 'test':
                overall_test_metrics = { k : event.data[k] for k in METRICS }
                overall_lower_bound_test_perplexity = event.data['lower_bound_perplexity']
    return Trial(
        lower_bound_dev_perplexity,
        dev_metrics,
        test_lengths,
        test_metrics,
        lower_bound_test_perplexities,
        overall_test_metrics,
        overall_lower_bound_test_perplexity
    )

def subtract_lower_bounds(trial, src, dest, f=lambda x: x):
    trial.dev_metrics[dest] = [
        f(x) - f(trial.lower_bound_dev_perplexity)
        for x in trial.dev_metrics[src]
    ]
    trial.test_metrics[dest] = [
        f(x) - f(y)
        for x, y in zip(
            trial.test_metrics[src],
            trial.lower_bound_test_perplexities
        )
    ]
    if trial.overall_test_metrics is not None:
        trial.overall_test_metrics[dest] = (
            f(trial.overall_test_metrics[src])
            - f(trial.overall_lower_bound_test_perplexity)
        )

def take_log(trial, src, dest):
    trial.dev_metrics[dest] = [math.log(x) for x in trial.dev_metrics[src]]
    trial.test_metrics[dest] = [math.log(x) for x in trial.test_metrics[src]]
    if trial.overall_test_metrics is not None:
        trial.overall_test_metrics[dest] = math.log(trial.overall_test_metrics[src])

def average_trials(trials, metric):
    data = transpose_trial_data(trials, metric)
    pairs = [
        average_array(numpy.array([y for y in x if y is not None]))
        for x in data
    ]
    return tuple(map(numpy.array, zip(*pairs)))

def transpose_trial_data(trials, metric):
    metrics = (x.dev_metrics[metric] for x in trials)
    metrics = [x for x in metrics if x]
    extend_lists(metrics)
    return zip(*metrics)

def extend_lists(seqs):
    n = max(map(len, seqs))
    for seq in seqs:
        fill = seq[-1]
        while len(seq) < n:
            seq.append(fill)

def average_array(x):
    return numpy.mean(x, axis=0), numpy.std(x, axis=0)

def truncate_lists(lists):
    min_len = min(map(len, lists))
    for x in lists:
        del x[min_len:]

def maybe_log(a, log_space):
    if log_space:
        numpy.log(a, out=a)

def force_integer_ticks(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5]))

def main():

    parser = argparse.ArgumentParser(
        description=
        'Generate plots of validation set performance during training.'
    )
    parser.add_argument('--inputs', type=pathlib.Path, nargs='+',
        help='Multiple input directories, one for each model being plotted. '
             'Each model directory should contain one or more trials. Each '
             'trial is a directory containing logs from training a model.')
    parser.add_argument('--labels', nargs='+', default=[],
        help='Labels for each model used in the plot. There should be as many '
             'labels as models.')
    parser.add_argument('--metric', default='cross-entropy-diff',
        help='Which metric to plot on the y-axis. Default is difference in '
             'cross entropy between the model and the lower bound on the '
             'validation set.')
    parser.add_argument('--title',
        help='Optional title to put over the plot.')
    parser.add_argument('--width', type=float, default=4.5,
        help='Width of the figure.')
    parser.add_argument('--height', type=float, default=3.5,
        help='Height of the figure.')
    parser.add_argument('--output', type=pathlib.Path, action='append', default=[],
        help='Output file. Format is controlled by the file extension.')
    parser.add_argument('--pgfplots-output', type=pathlib.Path,
        help='PGFPlots output file for LaTeX.')
    parser.add_argument('--show', action='store_true', default=False,
        help='Show the plot using Tk.')
    args = parser.parse_args()

    plt.rcParams.update({
        'font.family' : 'serif',
        'text.usetex' : False,
        'pgf.rcfonts' : False
    })
    fig, ax = plt.subplots()
    fig.set_size_inches(args.width, args.height)

    if args.title is not None:
        ax.set_title(args.title)
    if args.metric == 'cross-entropy-diff':
        ylabel = 'Difference in Cross Entropy'
    else:
        ylabel = args.metric.capitalize()
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Epoch')

    data = [read_data(d, args.metric) for d in args.inputs]
    x_lo = 1
    x_hi = max(len(z[0]) for z in data)

    for input_dir, (y, yerr), label in itertools.zip_longest(args.inputs, data, args.labels):
        if label is None:
            label = input_dir.name
        x = range(1, len(y)+1)
        line, = ax.plot(x, y, label=label)
        ax.fill_between(x, y-yerr, y+yerr, color=line.get_color(), alpha=0.2)

    force_integer_ticks(ax)
    ax.legend()
    plt.tight_layout()

    for output_path in args.output:
        plt.savefig(output_path)
    if args.pgfplots_output is not None:
        tikzplotlib.save(args.pgfplots_output)
    if args.show:
        plt.show()

if __name__ == '__main__':
    main()
