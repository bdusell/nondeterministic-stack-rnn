import argparse
import sys

import numpy

def file_to_numbers(fin):
    for line in fin:
        yield float(line.strip())

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--stddev', action='store_true', default=False)
    parser.add_argument('--precision', type=int)
    args = parser.parse_args()

    def format_float(x):
        if args.precision is not None:
            return f'{x:.{args.precision}f}'
        else:
            return str(x)

    values = numpy.array(list(file_to_numbers(sys.stdin)))
    mean = numpy.mean(values)
    if args.stddev:
        stddev = numpy.std(values)
        print(f'{format_float(mean)} $\\pm$ {format_float(stddev)}')
    else:
        print(format_float(mean))

if __name__ == '__main__':
    main()
