import argparse
import math
import pathlib

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import CenteredNorm
import tikzplotlib
import torch

from nsrnn.logging import read_log_file
from nsrnn.semiring import log
from plot_train import force_integer_ticks

def force_integer_ticks(axis, steps=[1, 2, 5]):
    axis.set_major_locator(MaxNLocator(integer=True, steps=steps))

def get_correct_row(op_weights):
    semiring = log
    n = len(op_weights)
    mid = (n - 1) // 2
    for t, ops_t in enumerate(op_weights):
        if t < mid:
            op_index = 0
        elif t < mid + 1:
            op_index = 1
        else:
            op_index = 2
        op_tensors = [torch.tensor(op[0]) for op in ops_t]
        op_sums = torch.stack([semiring.sum(op, dim=tuple(range(op.dim()))) for op in op_tensors])
        op_probs = torch.nn.functional.softmax(op_sums, dim=0)
        yield op_probs[op_index].item()

def get_total_row(op_weights):
    semiring = log
    for ops_t in op_weights:
        op_tensors = [torch.tensor(op[0]) for op in ops_t]
        # op_sums : 3 x [Q x S]
        op_sums = torch.stack([semiring.sum(op, dim=tuple(range(2, op.dim()))) for op in op_tensors])
        # op_sum : Q x S
        op_sum = semiring.sum(op_sums, dim=0)
        op_mean_sum = op_sum.mean()
        yield op_mean_sum.item()

TRAIN_DATA_SIZE = 10000
BATCH_SIZE = 10
VIS_UPDATES = 100

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pathlib.Path)
    parser.add_argument('--mode', choices=['correct', 'total'], default='correct')
    parser.add_argument('--rows-limit', type=int)
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

    UPDATES_PER_EPOCH = TRAIN_DATA_SIZE / BATCH_SIZE
    ROWS_PER_EPOCH = UPDATES_PER_EPOCH / VIS_UPDATES
    print('rows per epoch:', ROWS_PER_EPOCH)

    rows = []
    with args.input.open() as fin:
        for event in read_log_file(fin):
            if event.type == 'vis_data':
                sequences = event.data['data']
                sequence = sequences[0]
            elif event.type == 'signals':
                batch_elements = event.data['values']
                op_weights = batch_elements[0]
                if args.mode == 'correct':
                    row = get_correct_row(op_weights)
                else:
                    row = get_total_row(op_weights)
                rows.append(list(row))
                if len(rows) == args.rows_limit:
                    break

    if args.mode == 'total':
        min_value = min(x for row in rows for x in row)
        max_value = max(x for row in rows for x in row)
        print('min:', min_value, math.exp(min_value))
        print('max:', max_value, math.exp(max_value))

    print('plotting...')

    plt.rcParams.update({
        'font.family' : 'serif',
        'text.usetex' : False,
        'pgf.rcfonts' : False
    })
    fig, ax = plt.subplots()
    fig.set_size_inches(args.width, args.height)
    ax.set_xlabel('$t$')
    ax.set_ylabel('Epochs Elapsed')

    left = 0
    right = len(rows[0])
    top = 0
    bottom = len(rows) / ROWS_PER_EPOCH
    extent = (left, right, bottom, top)

    options = dict(
        aspect='auto',
        interpolation='none',
        extent=extent
    )

    if args.mode == 'correct':
        plt.imshow(rows, cmap='Greys', **options)
    else:
        plt.imshow(rows, norm=CenteredNorm(), cmap='RdYlGn', **options)

    force_integer_ticks(ax.yaxis, steps=[10])
    # Remove x ticks.
    ax.get_xaxis().set_ticks([])
    # Remove the black border.
    for k in ('top', 'right', 'bottom', 'left'):
        ax.spines[k].set_visible(False)

    plt.tight_layout()
    for output_path in args.output:
        plt.savefig(output_path)
    if args.pgfplots_output is not None:
        tikzplotlib.save(args.pgfplots_output)
    if args.show:
        plt.show()

if __name__ == '__main__':
    main()
