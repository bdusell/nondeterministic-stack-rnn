import argparse
import random

import torch

from utils.cli_util import parse_interval
from utils.data_util import generate_batches, construct_sampler
from utils.task_util import add_task_arguments, parse_task

def main():

    parser = argparse.ArgumentParser(
        description=
        'Generate and print data sampled for a particular task. Useful for '
        'debugging or tweaking the hyperparameters of the task.'
    )
    parser.add_argument('--data-seed', type=int,
        help='Random seed.')
    parser.add_argument('--length-range', type=parse_interval, required=True,
        help='The range of lengths that the generated sequenes will have.')
    parser.add_argument('--data-size', type=int, required=True,
        help='The number of sequences to generate.')
    parser.add_argument('--batch-size', type=int, default=1,
        help='Number of sequences per batch.')
    add_task_arguments(parser)
    args = parser.parse_args()

    device = torch.device('cpu')
    grammar, vocab = parse_task(parser, args)
    print(grammar)

    sampler = construct_sampler(grammar)
    vocab_size = vocab.size()
    generator = random.Random(args.data_seed)

    batches = generate_batches(
        sampler,
        sampler.valid_lengths(args.length_range),
        args.data_size,
        args.batch_size,
        vocab_size,
        generator,
        device
    )
    for x, y in batches:
        y_list = y.tolist()
        for y_elem in y_list:
            print(' '.join(map(vocab.value, y_elem)))

if __name__ == '__main__':
    main()
