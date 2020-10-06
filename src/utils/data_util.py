import random

import attr
import torch

from nsrnn.formal_models.pcfg_tools import (
    remove_epsilon_rules, remove_unary_rules)
from nsrnn.formal_models.pcfg_length_sampling import LengthSampler
from .cli_util import parse_interval

@attr.s
class Data:
    grammar = attr.ib()
    sampler = attr.ib()
    train_lengths = attr.ib()
    valid_lengths = attr.ib()
    train_data = attr.ib()
    valid_data = attr.ib()
    vocab = attr.ib()

def add_data_arguments(parser):
    group = parser.add_argument_group('Data generation options')
    group.add_argument('--data-seed', type=int,
        help='Random seed for generating training and validation data.')
    group.add_argument('--train-length-range', type=parse_interval, required=True,
        metavar='min:max',
        help='Range of lengths of training sequences generated.')
    group.add_argument('--train-data-size', type=int, required=True,
        help='Number of samples to generate for the training set.')
    group.add_argument('--batch-size', type=int, required=True,
        help='Batch size for the training set.')
    group.add_argument('--valid-length-range', type=parse_interval, required=True,
        metavar='min:max',
        help='Range of lengths of validation sequences generated.')
    group.add_argument('--valid-data-size', type=int, required=True,
        help='Number of samples to generate for the validation set.')
    group.add_argument('--valid-batch-size', type=int, required=True,
        help='Batch size for the validation set.')
    return group

def construct_sampler(grammar):
    # The length sampler requires epsilon and unary rules to be removed.
    remove_epsilon_rules(grammar)
    remove_unary_rules(grammar)
    return LengthSampler(grammar)

def generate_data(grammar, vocab, device, args):
    sampler = construct_sampler(grammar)
    vocab_size = vocab.size()
    generator = random.Random(args.data_seed)
    train_lengths = sampler.valid_lengths(args.train_length_range)
    valid_lengths = sampler.valid_lengths(args.valid_length_range)
    train_data = list(generate_batches(
        sampler,
        train_lengths,
        args.train_data_size,
        args.batch_size,
        vocab_size,
        generator,
        device))
    valid_data = list(generate_batches(
        sampler,
        valid_lengths,
        args.valid_data_size,
        args.valid_batch_size,
        vocab_size,
        generator,
        device))
    return Data(
        grammar,
        sampler,
        train_lengths,
        valid_lengths,
        train_data,
        valid_data,
        vocab
    )

def generate_batches(sampler, valid_lengths, num_samples, batch_size,
        vocab_size, generator, device):
    for actual_batch_size in compute_batch_sizes(num_samples, batch_size):
        length = generator.choice(valid_lengths)
        yield generate_batch(
            sampler,
            length,
            actual_batch_size,
            vocab_size,
            generator,
            device)

def compute_batch_sizes(size, batch_size):
    samples = 0
    while samples < size:
        actual_batch_size = min(batch_size, size - samples)
        yield actual_batch_size
        samples += actual_batch_size

def generate_batch(sampler, length, batch_size, vocab_size, generator, device):
    indexes = torch.tensor([
        list(sampler.sample(length, generator))
        for i in range(batch_size)
    ], dtype=torch.long, device=device)
    x = indexes_to_one_hot_input_tensor(indexes, vocab_size)
    # The output includes all elements of the sequence, including the first,
    # because we do want to assign a probability to the first element of the
    # sequence.
    return x, indexes

def indexes_to_one_hot_input_tensor(indexes, vocab_size):
    # The input includes all elements of the sequence except the last.
    input_tensor_indexes = indexes[:, :-1]
    input_tensor = torch.zeros(
        input_tensor_indexes.size() + (vocab_size,),
        device=indexes.device
    )
    if input_tensor.size(1) > 0:
        input_tensor.scatter_(2, input_tensor_indexes.unsqueeze(2), 1)
    return input_tensor

def batches_to_sequences(batches):
    for x, y in batches:
        for y_elem in y.tolist():
            yield y_elem
