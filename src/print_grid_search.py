import argparse
import math
import pathlib
import sys

import attr
import numpy

from nsrnn.pytorch_tools.saver import read_logs

def read_data(dirname, trials, ignore_missing_trials):
    settings = (
        read_data_for_setting(d, trials, ignore_missing_trials)
        for d in sort_alphabetically(dirname.iterdir())
    )
    return [setting for setting in settings if setting is not None]

@attr.s
class Setting:
    path = attr.ib()
    hyperparams = attr.ib()
    metrics = attr.ib()

def read_data_for_setting(dirname, trials, ignore_missing_trials):
    trial_dirs = [dirname / str(i+1) for i in range(trials)]
    trials = [read_data_for_trial(d) for d in trial_dirs]
    for trial_dir, trial in zip(trial_dirs, trials):
        if trial is None:
            if ignore_missing_trials:
                print(f'trial {trial_dir} is missing', file=sys.stderr)
            else:
                raise RuntimeError(f'trial {trial_dir} is missing')
    trials = [trial for trial in trials if trial is not None]
    if trials:
        hyperparams = trials[0].hyperparams
        averaged_metrics = average_dicts([trial.metrics for trial in trials])
        return Setting(
            dirname,
            hyperparams,
            averaged_metrics
        )
    else:
        return None

@attr.s
class Trial:
    hyperparams = attr.ib()
    metrics = attr.ib()

def read_data_for_trial(dirname):
    start_data = None
    metrics = None
    try:
        with read_logs(dirname) as events:
            for event in events:
                if event.type == 'start':
                    start_data = event.data
                elif event.type == 'train':
                    src = event.data['best_validation_metrics']
                    metrics = { k : src[k] for k in ('accuracy', 'perplexity') }
                    metrics['epochs_since_improvement'] = event.data['epochs_since_improvement']
    except FileNotFoundError:
        pass
    if start_data is None or metrics is None:
        return None
    cross_entropy = math.log(metrics['perplexity'])
    cross_entropy_lower_bound = math.log(start_data['valid_lower_bound_perplexity'])
    cross_entropy_diff = cross_entropy - cross_entropy_lower_bound
    metrics['cross_entropy'] = cross_entropy
    metrics['cross_entropy_diff'] = cross_entropy_diff
    return Trial(
        start_data,
        metrics
    )

def sort_alphabetically(paths):
    return sorted(paths, key=lambda x: x.name)

def average_dicts(dicts):
    d = aggregate_dicts(dicts)
    return { k : average_array(numpy.array(v)) for k, v in d.items() }

def average_array(x):
    return numpy.mean(x, axis=0), numpy.std(x, axis=0)

def aggregate_dicts(dicts):
    result = {}
    keys = None
    initial = True
    for d in dicts:
        if initial:
            keys = d.keys()
            for k in keys:
                result[k] = []
            initial = False
        else:
            if d.keys() != keys:
                raise ValueError
        for k, v in d.items():
            result[k].append(v)
    return result

def print_setting(setting):
    perplexity = average_to_str(setting.metrics['perplexity'])
    cross_entropy_diff = average_to_str(setting.metrics['cross_entropy_diff'], 3)
    accuracy = average_to_str(setting.metrics['accuracy'])
    print(
        f'perplexity: {perplexity} | '
        f'cross entropy diff: {cross_entropy_diff} | '
        f'accuracy: {accuracy} || '
        f'dir name: {setting.path.name}'
    )

def average_to_str(pair, sig=2):
    mean, std = pair
    return f'{mean:.{sig}f} ± {std:.{sig}f}'

def main():

    parser = argparse.ArgumentParser(
        description=
        'For a given model, print the hyperparameter setting with the best '
        'average performance on the validation set.'
    )
    parser.add_argument('directory', type=pathlib.Path,
        help='A directory containing one or more hyperparameter directories. '
             'Each hyperparameter directory should contain n trial '
             'directories numbered 1 through n. Each trial directory '
             'should be a directory containing logs from training a model. '
             'This script prints the path to the hyperparameter directory '
             'with the best performance.')
    parser.add_argument('--metric', default='cross_entropy_diff',
        help='Which metric to use to determine the best setting. Default is '
             'difference in cross entropy between the model and the lower '
             'bound on the validation set.')
    parser.add_argument('--trials', type=int, required=True,
        help='The number of trials to expect within each hyperparameter '
             'directory. Missing trials will be treated as errors.')
    parser.add_argument('--ignore-missing-trials', action='store_true', default=False,
        help='Don\'t treat missing trials as errors, but print warnings '
             'instead.')
    args = parser.parse_args()

    data = read_data(args.directory, args.trials, args.ignore_missing_trials)
    # NOTE: This sorts by lowest mean, then lowest std dev.
    best_setting = min(data, key=lambda x: x.metrics[args.metric])
    print(best_setting.path)

if __name__ == '__main__':
    main()
