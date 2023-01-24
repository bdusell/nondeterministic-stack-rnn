import argparse
import numpy
import pathlib
import sys

from lib.pytorch_tools.saver import read_logs
from lib.logging import LogParseError
from utils.print_best import read_data_for_multiple_trials

def read_test_data(dirname):
    metrics = None
    try:
        with read_logs(dirname) as events:
            for event in events:
                if event.type == 'test':
                    metrics = event.data
    except (FileNotFoundError, LogParseError):
        pass
    return metrics

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--label', action='append', default=[])
    parser.add_argument('--inputs', type=pathlib.Path, nargs='*', action='append', default=[])
    parser.add_argument('--show-mean', action='store_true', default=False)
    args = parser.parse_args()

    labels = args.label
    input_lists = args.inputs
    if len(labels) != len(input_lists):
        parser.error('must have the same number of --label and --input arguments')

    target_runs = max(len(input_list) for input_list in input_lists)
    labels_and_trials = []
    all_missing_dirs = []
    for label, input_list in zip(labels, input_lists):
        trials, missing_dirs = read_data_for_multiple_trials(input_list)
        labels_and_trials.append((label, trials))
        all_missing_dirs.extend(missing_dirs)
    show_runs = not all(len(trials) == target_runs for label, trials in labels_and_trials)

    columns = []
    column_specs = []
    columns.append('model')
    column_specs.append('l')
    if args.show_mean:
        columns.append('mean_val')
        column_specs.append('c')
    columns.append('min_val')
    column_specs.append('c')
    columns.append('min_test')
    column_specs.append('c')
    if show_runs:
        columns.append('runs')
        column_specs.append('c')
    columns_set = set(columns)
    column_spec = ''.join(column_specs)

    print(f'\\begin{{tabular}}{{@{{}}{column_spec}@{{}}}}')
    print('\\toprule')

    headings = []
    if 'model' in columns_set:
        headings.append('Model')
    if 'mean_val' in columns_set:
        headings.append('Mean Val.')
    if 'min_val' in columns_set:
        headings.append('Val.' if 'mean_val' not in columns_set else 'Min. Val.')
    if 'min_test' in columns_set:
        headings.append('Test')
    if 'runs' in columns_set:
        headings.append('Runs')
    print(' & '.join(headings) + ' \\\\')
    print('\\midrule')

    rows = []
    for label, trials in labels_and_trials:
        row = { 'model' : label }
        if trials:
            validation_perplexity_array = numpy.array([
                trial.metrics['perplexity']
                for trial in trials
            ])
            if 'mean_val' in columns_set:
                row['mean_val'] = numpy.mean(validation_perplexity_array)
                row['std_val'] = numpy.std(validation_perplexity_array)
            if 'min_val' in columns_set or 'min_test' in columns_set:
                argmin_val = numpy.argmin(validation_perplexity_array)
            if 'min_val' in columns_set:
                row['min_val'] = validation_perplexity_array[argmin_val]
            if 'min_test' in columns_set:
                best_trial = trials[argmin_val]
                test_metrics = read_test_data(best_trial.path / 'test')
                if test_metrics is not None:
                    row['min_test'] = test_metrics['perplexity']
            if 'runs' in columns_set:
                row['runs'] = len(trials)
        rows.append(row)

    if 'min_val' in columns_set:
        best_min_val = min(row['min_val'] for row in rows if 'min_val' in row)
    if 'min_test' in columns_set:
        best_min_test = min(row['min_test'] for row in rows if 'min_test' in row)
    for row in rows:
        table_row = []
        if 'model' in columns_set:
            table_row.append(row['model'])
        if 'mean_val' in columns_set:
            if 'mean_val' in row:
                mean_val = row['mean_val']
                std_val = row['std_val']
                table_row.append(f'${mean_val:.2f} \\pm {val_std:.2f}$')
            else:
                table_row.append('')
        if 'min_val' in columns_set:
            if 'min_val' in row:
                min_val = row['min_val']
                cell = f'{min_val:.2f}'
                if min_val == best_min_val:
                    cell = f'\\textbf{{{cell}}}'
                table_row.append(cell)
            else:
                table_row.append('')
        if 'min_test' in columns_set:
            if 'min_test' in row:
                min_test = row['min_test']
                cell = f'{min_test:.2f}'
                if min_test == best_min_test:
                    cell = f'\\textbf{{{cell}}}'
                table_row.append(cell)
            else:
                table_row.append('')
        if 'runs' in columns_set:
            table_row.append(str(row['runs']))
        print(' & '.join(table_row) + ' \\\\')

    print('\\bottomrule')
    print('\\end{tabular}')
    if show_runs:
        print(f'% info: results are not complete (targeting {target_runs} runs)')
    else:
        print(f'% info: all results are complete and are aggregated from {target_runs} runs')
    for missing_dir in all_missing_dirs:
        print(f'% missing: {missing_dir}', file=sys.stderr)

if __name__ == '__main__':
    main()
