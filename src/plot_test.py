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

@attr.s
class Data:
    settings = attr.ib()
    test_lengths = attr.ib()
    lower_bound_test_perplexities = attr.ib()
    overall_lower_bound_test_perplexity = attr.ib()

def read_data(dirnames):
    settings = [read_data_for_setting(x) for x in dirnames]
    first_setting = settings[0]
    other_settings = settings[1:]
    for setting in other_settings:
        if setting.test_lengths != first_setting.test_lengths:
            raise ValueError(
                f'settings have mismatched test lengths:\n'
                f'{first_setting.dirname}: {first_setting.test_lengths}\n'
                f'{setting.dirname}: {setting.test_lengths}')
        if setting.lower_bound_test_perplexities != first_setting.lower_bound_test_perplexities:
            raise ValueError(
                f'settings have mismatched lower bound test perplexities:\n'
                f'{first_setting.dirname}: {first_setting.lower_bound_test_perplexities}\n'
                f'{setting.dirname}: {setting.lower_bound_test_perplexities}')
        if setting.overall_lower_bound_test_perplexity != first_setting.overall_lower_bound_test_perplexity:
            raise ValueError(
                f'settings have mismatched lower bound test perplexities:\n'
                f'{first_setting.dirname}: {first_setting.overall_lower_bound_test_perplexity}\n'
                f'{setting.dirname}: {setting.overall_lower_bound_test_perplexity}')
    return Data(
        settings,
        first_setting.test_lengths,
        first_setting.lower_bound_test_perplexities,
        first_setting.overall_lower_bound_test_perplexity
    )

def read_data_for_setting(dirname):
    trials = [read_data_for_trial(d) for i, d in list_dirs_in_order(dirname)]
    return average_trials(dirname, trials)

def list_dirs_in_order(dirname):
    pairs = ((int(d.name), d) for d in dirname.iterdir())
    return sorted(pairs, key=lambda x: x[0])

@attr.s
class Trial:
    dirname = attr.ib()
    test_lengths = attr.ib()
    test_metrics = attr.ib()
    lower_bound_test_perplexities = attr.ib()
    overall_test_metrics = attr.ib()
    overall_lower_bound_test_perplexity = attr.ib()

METRICS = ('perplexity', 'accuracy')

def read_data_for_trial(dirname):
    test_lengths = []
    test_metrics = { k : [] for k in METRICS }
    test_metrics['cross_entropy_diff'] = []
    lower_bound_test_perplexities = []
    overall_test_metrics = None
    overall_lower_bound_test_perplexity = None
    with read_logs(dirname) as events:
        events = list(events)
        num_valid_lengths = sum(int(e.type == 'test_length') for e in events)
        for event in events:
            if event.type == 'test_length':
                test_lengths.append(event.data['length'])
                for k in METRICS:
                    test_metrics[k].append(event.data[k])
                perplexity = event.data['perplexity']
                lower_bound_perplexity = event.data['lower_bound_perplexity']
                lower_bound_test_perplexities.append(lower_bound_perplexity)
                cross_entropy_diff = math.log(perplexity) - math.log(lower_bound_perplexity)
                test_metrics['cross_entropy_diff'].append(cross_entropy_diff)
            elif event.type == 'test':
                overall_test_metrics = { k : event.data[k] for k in METRICS }
                overall_lower_bound_test_perplexity = event.data['lower_bound_perplexity']
    return Trial(
        dirname,
        test_lengths,
        test_metrics,
        lower_bound_test_perplexities,
        overall_test_metrics,
        overall_lower_bound_test_perplexity
    )

@attr.s
class Setting:
    dirname = attr.ib()
    test_lengths = attr.ib()
    test_cross_entropy_diff = attr.ib()
    lower_bound_test_perplexities = attr.ib()
    overall_test_perplexity = attr.ib()
    overall_lower_bound_test_perplexity = attr.ib()

def average_trials(dirname, trials):
    first_trial = trials[0]
    other_trials = trials[1:]
    for trial in other_trials:
        if trial.test_lengths != first_trial.test_lengths:
            raise ValueError(
                f'trials have mismatched test lengths:\n'
                f'{first_trial.dirname}: {first_trial.test_lengths}\n'
                f'{trial.dirname}: {trial.test_lengths}')
        if trial.lower_bound_test_perplexities != first_trial.lower_bound_test_perplexities:
            raise ValueError(
                f'trials have mismatched lower bound test perplexities:\n'
                f'{first_trial.dirname}: {first_trial.lower_bound_test_perplexities}\n'
                f'{trial.dirname}: {trial.lower_bound_test_perplexities}')
        if trial.overall_lower_bound_test_perplexity != first_trial.overall_lower_bound_test_perplexity:
            raise ValueError(
                f'trials have mismatched lower bound test perplexities:\n'
                f'{first_trial.dirname}: {first_trial.overall_lower_bound_test_perplexity}\n'
                f'{trial.dirname}: {trial.overall_lower_bound_test_perplexity}')
    metric = 'cross_entropy_diff'
    test_metrics = average_array([x.test_metrics[metric] for x in trials])
    overall_test_metrics = None # average_array([x.overall_test_metrics[metric] for x in trials])
    return Setting(
        dirname,
        first_trial.test_lengths,
        test_metrics,
        first_trial.lower_bound_test_perplexities,
        overall_test_metrics,
        first_trial.overall_lower_bound_test_perplexity
    )

def average_array(x):
    return numpy.mean(x, axis=0), numpy.std(x, axis=0)

def force_integer_ticks(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5]))

def main():

    parser = argparse.ArgumentParser(
        description=
        'Generate plots of test set performance.'
    )
    parser.add_argument('--inputs', type=pathlib.Path, nargs='+',
        help='Multiple input directories, one for each model being plotted. '
             'Each model directory should contain one or more trials. Each '
             'trial is a directory containing logs from evaluating on the '
             'test set.')
    parser.add_argument('--labels', nargs='+', default=[],
        help='Labels for each model used in the plot. There should be as many '
             'labels as models.')
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
    ax.set_xlabel('Length')

    data = read_data(args.inputs)
    x = data.test_lengths
    for setting, label in itertools.zip_longest(data.settings, args.labels):
        if label is None:
            label = setting.dirname
        y, yerr = setting.test_cross_entropy_diff
        line, = ax.plot(x, y, label=label)
        ax.fill_between(x, y-yerr, y+yerr, color=line.get_color(), alpha=0.2)

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
