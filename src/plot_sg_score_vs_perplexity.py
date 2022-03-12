import argparse
import csv
import itertools
import pathlib

import matplotlib.pyplot as plt
import tikzplotlib

def read_data(path):
    with path.open() as fin:
        x, y = zip(*(map(float, row) for row in csv.reader(fin, delimiter='\t')))
    return x, y

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=pathlib.Path, nargs='+')
    parser.add_argument('--labels', nargs='+', default=[])
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

    ax.set_ylabel('SG Score')
    ax.set_xlabel('Perplexity')

    for input_path, label in itertools.zip_longest(args.inputs, args.labels):
        if label is None:
            label = input_path.name
        x, y = read_data(input_path)
        ax.plot(x, y, '.', label=label)

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
