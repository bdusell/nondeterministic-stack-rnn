import itertools
import random

import attr
import torch

from utils.cli_util import parse_interval

@attr.s
class Data:
    sampler = attr.ib()
    train_lengths = attr.ib()
    valid_lengths = attr.ib()
    train_data = attr.ib()
    valid_data = attr.ib()
    input_vocab = attr.ib()
    output_vocab = attr.ib()
    random_seed = attr.ib()

def add_data_arguments(parser):
    group = parser.add_argument_group('Data generation options')
    group.add_argument('--data-seed', type=int,
        help='Random seed for generating training and validation data.')
    group.add_argument('--train-length-range', type=parse_interval,
        metavar='min:max',
        default=(40, 80),
        help='Range of lengths of training sequences generated.')
    group.add_argument('--train-data-size', type=int,
        default=10000,
        help='Number of samples to generate for the training set.')
    group.add_argument('--batch-size', type=int,
        default=10,
        help='Batch size for the training set.')
    group.add_argument('--valid-length-range', type=parse_interval,
        metavar='min:max',
        default=(40, 80),
        help='Range of lengths of validation sequences generated.')
    group.add_argument('--valid-data-size', type=int,
        default=1000,
        help='Number of samples to generate for the validation set.')
    group.add_argument('--valid-batch-size', type=int,
        default=10,
        help='Batch size for the validation set.')
    return group

def get_random_generator_and_seed(random_seed):
    if random_seed is None:
        random_seed = random.getrandbits(32)
    return random.Random(random_seed), random_seed

def get_random_generator(random_seed):
    generator, random_seed = get_random_generator_and_seed(random_seed)
    return generator

def generate_data(task, device, args):
    sampler = task.sampler
    input_vocab_size = task.input_vocab.size()
    generator, random_seed = get_random_generator_and_seed(args.data_seed)
    train_lengths = sampler.valid_lengths(*args.train_length_range)
    valid_lengths = sampler.valid_lengths(*args.valid_length_range)
    train_data = list(generate_batches(
        sampler,
        train_lengths,
        args.train_data_size,
        args.batch_size,
        input_vocab_size,
        generator,
        device))
    valid_data = list(generate_batches(
        sampler,
        valid_lengths,
        args.valid_data_size,
        args.valid_batch_size,
        input_vocab_size,
        generator,
        device))
    return Data(
        sampler,
        train_lengths,
        valid_lengths,
        train_data,
        valid_data,
        task.input_vocab,
        task.output_vocab,
        random_seed
    )

def generate_batches(sampler, valid_lengths, num_samples, batch_size,
        input_vocab_size, generator, device, input_includes_eos=False):
    """Generate batches of data.

    :param sampler: A data sampler.
    :param valid_lengths: A list of the lengths for which sequences will be
        sampled.
    :param num_samples: The number of samples to return.
    :param batch_size: The size of each returned batch. The last batch will
        have fewer samples if num_samples is not divisible by batch_size.
    :param input_vocab_size: The number of symbol types in the input
        vocabulary.
    :param generator: A random number generator.
    :param device: The PyTorch device where the tensors will be created.
    :return: An iterable of pairs of tensors.
    """
    batch_sizes_and_lengths = (
        (actual_batch_size, generator.choice(valid_lengths))
        for actual_batch_size in compute_batch_sizes(num_samples, batch_size)
    )
    return generate_batches_with_sizes(
        sampler,
        batch_sizes_and_lengths,
        input_vocab_size,
        generator,
        device,
        input_includes_eos
    )

def generate_batches_with_sizes(sampler, batch_sizes_and_lengths,
        input_vocab_size, generator, device, input_includes_eos=False):
    for batch_size, length in batch_sizes_and_lengths:
        yield generate_batch(
            sampler,
            length,
            batch_size,
            input_vocab_size,
            generator,
            device,
            input_includes_eos
        )

def compute_batch_sizes(size, batch_size):
    samples = 0
    while samples < size:
        actual_batch_size = min(batch_size, size - samples)
        yield actual_batch_size
        samples += actual_batch_size

def generate_batch(sampler, length, batch_size, input_vocab_size, generator,
        device, input_includes_eos=False):
    """Generate a single batch of data as a pair of tensors (x, y).

    All sequences in this batch will have exactly the same length. There is no
    padding symbol.

    :param length: The length of all the sequences in the batch.
    :param batch_size: The number of samples in the batch.
    :param input_vocab_size: The number of symbol types in the input
        vocabulary.
    :param generator: A random number generator.
    :param device: The PyTorch device where the tensors will be created.
    :return: A pair of tensors.
    """
    # Make sure every sequence ends with an end-of-sequence symbol.
    eos = (input_vocab_size,)
    indexes = torch.tensor([
        list(itertools.chain(sampler.sample(length, generator), eos))
        for i in range(batch_size)
    ], dtype=torch.long, device=device)
    if input_includes_eos:
        input_vocab_size += 1
    x = indexes_to_one_hot_input_tensor(indexes, input_vocab_size)
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

def batches_to_sequences(batches, include_eos=False):
    for x, y in batches:
        for y_elem in y.tolist():
            if not include_eos:
                # Remove the end-of-sequence symbol.
                y_elem = y_elem[:-1]
            yield y_elem
