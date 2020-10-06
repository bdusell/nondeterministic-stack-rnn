import argparse
import logging
import pathlib
import random
import sys

import torch

from nsrnn.lang_algorithm.parsing import Parser
from nsrnn.lower_bound_perplexity import (
    compute_lower_bound_parts, parts_to_perplexity)
from utils.cli_util import parse_interval
from utils.data_util import (
    construct_sampler, generate_batches, batches_to_sequences)
from utils.task_util import add_task_arguments, parse_task

def main():

    argv = sys.argv[1:]
    logger = logging.getLogger('main')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description=
        'Generate a set of test data for a particular task, compute its '
        'lower bound perplexity, and save that information to a .pt file. '
        'Models can later be evaluated on the saved test set.'
    )
    parser.add_argument('--test-length-range', type=parse_interval, required=True,
        metavar='min:max',
        help='The range of lengths that the generated sequences will have.')
    parser.add_argument('--test-data-size', type=int, required=True,
        help='The number of sequences to generate per length.')
    parser.add_argument('--test-batch-size', type=int, default=256,
        help='Number of sequences per batch.')
    parser.add_argument('--test-data-seed', type=int,
        help='Random seed.')
    parser.add_argument('--output', type=pathlib.Path,
        help='Output file in .pt format.')
    add_task_arguments(parser)
    args = parser.parse_args()

    device = torch.device('cpu')
    grammar, vocab = parse_task(parser, args)
    sampler = construct_sampler(grammar)
    parser = Parser(grammar)

    test_data_generator = random.Random()
    test_data_generator.seed(args.test_data_seed)
    test_perp_numer = 0.0
    test_acc_numer = 0
    test_num_symbols = 0
    test_num_samples = 0
    test_lower_perp_numer = 0.0
    valid_lengths = sampler.valid_lengths(args.test_length_range)
    rows = []
    for length in valid_lengths:
        logger.info(f'length {length}')
        row = {}
        row['length'] = length
        batches = list(generate_batches(
            sampler=sampler,
            valid_lengths=[length],
            num_samples=args.test_data_size,
            batch_size=args.test_batch_size,
            vocab_size=vocab.size(),
            generator=test_data_generator,
            device=device
        ))
        row['batches'] = batches
        parts = compute_lower_bound_parts(
            sampler=sampler,
            parser=parser,
            samples=batches_to_sequences(batches))
        length_lower_perp_numer, length_num_symbols, length_num_samples = parts
        length_lower_perp = parts_to_perplexity(parts, 1)
        logger.info(f'  lower bound perplexity: {length_lower_perp:.3f}')
        row['lower_bound_perplexity_numerator'] = length_lower_perp_numer
        row['num_symbols'] = length_num_symbols
        row['num_samples'] = length_num_samples
        row['lower_bound_perplexity'] = length_lower_perp
        test_num_symbols += length_num_symbols
        test_num_samples += length_num_samples
        test_lower_perp_numer += length_lower_perp_numer
        rows.append(row)
    test_parts = (test_lower_perp_numer, test_num_symbols, test_num_samples)
    test_lower_perp = parts_to_perplexity(test_parts, len(valid_lengths))
    logger.info(f'test lower bound perplexity: {test_lower_perp:.3f}')
    if args.output is not None:
        torch.save({
            'lengths' : rows,
            'total' : {
                'lower_bound_perplexity_numerator' : test_lower_perp_numer,
                'num_symbols' : test_num_symbols,
                'num_samples' : test_num_samples,
                'lower_bound_perplexity' : test_lower_perp
            }
        }, args.output)

if __name__ == '__main__':
    main()
