import argparse
import itertools
import math
import pathlib

import attr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tikzplotlib

from nsrnn.pytorch_tools.saver import read_logs

@attr.s
class Setting:
    best_trial = attr.ib()

def read_data(dirname, metric, single):
    if single:
        trials = [read_data_for_trial(dirname, metric)]
    else:
        trials = [read_data_for_trial(d, metric) for i, d in list_dirs_in_order(dirname)]
    if metric == 'cross_entropy_diff':
        for trial in trials:
            subtract_lower_bounds(trial, 'perplexity', 'cross_entropy_diff', math.log)
    best_trial = min(trials, key=lambda x: x.best_valid_metrics[metric])
    return Setting(best_trial)

def list_dirs_in_order(dirname):
    pairs = ((int(d.name), d) for d in dirname.iterdir())
    return sorted(pairs, key=lambda x: x[0])

@attr.s
class Trial:
    lower_bound_valid_perplexity = attr.ib()
    valid_metrics = attr.ib()
    best_valid_metrics = attr.ib()

METRICS = ['perplexity']

def read_data_for_trial(dirname, metric):
    lower_bound_valid_perplexity = None
    valid_metrics = { k : [] for k in METRICS }
    best_valid_metrics = None
    with read_logs(dirname) as events:
        for event in events:
            if event.type == 'start':
                lower_bound_valid_perplexity = event.data['valid_lower_bound_perplexity']
            elif event.type == 'validate':
                for k in METRICS:
                    valid_metrics[k].append(event.data[k])
            elif event.type == 'train':
                best_valid_metrics = { k : event.data['best_validation_metrics'][k] for k in METRICS }
    return Trial(
        lower_bound_valid_perplexity,
        valid_metrics,
        best_valid_metrics
    )

def subtract_lower_bounds(trial, src, dest, f=lambda x: x):
    trial.valid_metrics[dest] = [
        f(x) - f(trial.lower_bound_valid_perplexity)
        for x in trial.valid_metrics[src]
    ]
    trial.best_valid_metrics[dest] = (
        f(trial.best_valid_metrics[src]) -
        f(trial.lower_bound_valid_perplexity)
    )

def force_integer_ticks(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5]))

def main():

    parser = argparse.ArgumentParser(
        description=
        'Generate plots of validation set performance during training.'
    )
    parser.add_argument('--inputs', type=pathlib.Path, nargs='+', default=[],
        help='Multiple input directories, one for each model being plotted. '
             'Each model directory should contain one or more trials. Each '
             'trial is a directory containing logs from training a model.')
    parser.add_argument('--labels', nargs='+', default=[],
        help='Labels for each model used in the plot. There should be as many '
             'labels as models.')
    parser.add_argument('--single', action='store_true', default=False,
        help='Do not treat inputs as directories containing multiple random '
             'restarts, but single runs.')
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
    ax.set_ylabel('Difference in Cross Entropy')
    ax.set_xlabel('Epoch')

    data = (read_data(d, 'cross_entropy_diff', args.single) for d in args.inputs)

    for input_dir, setting, label in itertools.zip_longest(args.inputs, data, args.labels):
        if label is None:
            label = input_dir.name
        ymin = setting.best_trial.valid_metrics['cross_entropy_diff']
        x = range(1, len(ymin)+1)
        line, = ax.plot(x, ymin, label=label)

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
