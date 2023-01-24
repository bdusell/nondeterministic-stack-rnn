import argparse
import random

from utils.cli_util import parse_interval
from cfl_language_modeling.grammars.hardest_cfl import HardestCFLGrammar
from cfl_language_modeling.data_util import construct_sampler

START_CONTENT = '\033[32m'
END_CONTENT = '\033[0m'

def tree_to_marked_string(tree):
    g = HardestCFLGrammar

    def recurse(tree):
        if tree.symbol.is_terminal:
            yield g.int_to_str(tree.symbol.value)
        else:
            if tree.symbol == g.Q:
                yield END_CONTENT
            for child in tree.children:
                for s in recurse(child):
                    yield s
            if tree.symbol == g.Q:
                yield START_CONTENT

    in_content = False
    for child in tree.children:
        if child.symbol in (g.S, g.T, g.dollar):
            if not in_content:
                yield START_CONTENT
            in_content = True
        else:
            if in_content:
                yield END_CONTENT
            in_content = False
        for s in recurse(child):
            yield s

def main():

    parser = argparse.ArgumentParser(
        description=
        'Sample and print strings from the Hardest CFL. The pieces of the '
        'true Dyck string are highlighted to distinguish them from the decoys.'
    )
    parser.add_argument('--samples', type=int, default=1,
        help='The number of samples to print.')
    parser.add_argument('--length', type=parse_interval, required=True,
        metavar='min:max',
        help='The range of lengths of strings to generate.')
    parser.add_argument('--mean-num-commas', type=float, required=True)
    parser.add_argument('--mean-short-filler-length', type=float, required=True)
    parser.add_argument('--mean-long-filler-length', type=float, required=True)
    parser.add_argument('--semicolon-probability', type=float, required=True)
    parser.add_argument('--mean-bracket-splits', type=float, required=True)
    parser.add_argument('--mean-nesting-depth', type=float, required=True)
    args = parser.parse_args()

    G = HardestCFLGrammar(
        mean_num_commas=args.mean_num_commas,
        mean_short_filler_length=args.mean_short_filler_length,
        mean_long_filler_length=args.mean_long_filler_length,
        semicolon_probability=args.semicolon_probability,
        mean_bracket_splits=args.mean_bracket_splits,
        mean_nesting_depth=args.mean_nesting_depth
    )
    sampler = construct_sampler(G)
    generator = random.Random()
    for i in range(args.samples):
        length = generator.randint(*args.length)
        tree = sampler.sample_tree(length)
        s = tree_to_marked_string(tree)
        print(''.join(s))

if __name__ == '__main__':
    main()
