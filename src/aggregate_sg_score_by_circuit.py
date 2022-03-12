import argparse
import csv
import json
import pathlib

import attr
import numpy

@attr.s
class Circuit:
    name = attr.ib()
    suites = attr.ib()

def read_circuits(path):
    with path.open() as fin:
        data = json.load(fin)
    return [
        Circuit(**circuit)
        for circuit in data['circuits']
    ]

def read_scores(path):
    with path.open() as fin:
        for name, score in csv.reader(fin, delimiter='\t', quoting=csv.QUOTE_NONE):
            yield name, float(score)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('scores', type=pathlib.Path)
    parser.add_argument('--circuits', type=pathlib.Path, required=True)
    args = parser.parse_args()

    scores = dict(read_scores(args.scores))
    circuits = read_circuits(args.circuits)
    circuit_scores = [
        numpy.mean([scores.pop(suite) for suite in circuit.suites])
        for circuit in circuits
    ]
    if scores:
        raise ValueError(
            f'the scores file included extra scores not in any circuit: '
            f'{", ".join(scores.keys())}')
    print('\t'.join(c.name for c in circuits))
    print('\t'.join(f'{score:.3f}' for score in circuit_scores))

if __name__ == '__main__':
    main()
