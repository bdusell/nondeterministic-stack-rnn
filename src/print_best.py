import argparse
import math
import pathlib
import sys

import attr

from nsrnn.pytorch_tools.saver import read_logs
from nsrnn.logging import LogParseError

@attr.s
class Trial:
    hyperparams = attr.ib()
    metrics = attr.ib()
    path = attr.ib()

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
                    metrics = { k : src[k] for k in ['perplexity'] }
    except (FileNotFoundError, LogParseError):
        pass
    if start_data is None or metrics is None:
        return None
    cross_entropy = math.log(metrics['perplexity'])
    metrics['cross_entropy'] = cross_entropy
    if 'valid_lower_bound_perplexity' in start_data:
        cross_entropy_lower_bound = math.log(start_data['valid_lower_bound_perplexity'])
        cross_entropy_diff = cross_entropy - cross_entropy_lower_bound
        metrics['cross_entropy_diff'] = cross_entropy_diff
    return Trial(start_data, metrics, dirname)

def read_data_for_multiple_trials(trial_dirs, ignore_missing_trials):
    trials = [read_data_for_trial(d) for d in trial_dirs]
    for trial_dir, trial in zip(trial_dirs, trials):
        if trial is None:
            if ignore_missing_trials:
                print(f'warning: data for trial {trial_dir} is missing', file=sys.stderr)
            else:
                raise RuntimeError(f'data for trial {trial_dir} is missing')
    return [trial for trial in trials if trial is not None]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=pathlib.Path)
    parser.add_argument('--ignore-missing-trials', action='store_true', default=False)
    parser.add_argument('--metric', default='cross_entropy_diff')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    trials = read_data_for_multiple_trials(args.inputs, args.ignore_missing_trials)
    key = lambda x: x.metrics[args.metric]
    if args.verbose:
        trials.sort(key=key)
        for trial in trials:
            print(trial)
    else:
        best_trial = min(trials, key=key)
        print(best_trial.path)

if __name__ == '__main__':
    main()
