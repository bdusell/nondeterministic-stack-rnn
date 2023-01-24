import math
import pathlib

import attr
import more_itertools
import torch

from .build_vocab import load_vocab

@attr.s
class Data:
    train_data = attr.ib()
    valid_data = attr.ib()
    test_data = attr.ib()
    vocab = attr.ib()

@attr.s
class DataSection:
    tensor_pairs = attr.ib()
    padding_length = attr.ib()

def add_natural_data_arguments(parser, train=False, valid=False, test=False,
        require_vocab=True):
    group = parser.add_argument_group('Data loading options')
    if train:
        group.add_argument('--train-data', type=pathlib.Path, required=True)
        group.add_argument('--batch-size', type=int, required=True)
        group.add_argument('--bptt-limit', type=int, required=True)
    if valid:
        group.add_argument('--valid-data', type=pathlib.Path, required=True)
    if test:
        group.add_argument('--test-data', type=pathlib.Path, required=True)
    if valid or test:
        group.add_argument('--eval-batch-size', type=int, required=True)
        group.add_argument('--eval-bptt-limit', type=int, required=True)
    group.add_argument('--vocab', type=pathlib.Path, required=require_vocab)
    if train or valid or test:
        group.add_argument('--truncate-data', action='store_true', default=False,
            help='Ensure that all batches are of equal length and batch size by '
                 'truncating the data.')
    return group

def load_data(device, args, train=False, valid=False, test=False, include_vocab=True):
    if include_vocab or train or valid or test:
        _, vocab = load_vocab(args.vocab)
    else:
        vocab = None
    if train or valid or test:
        kwargs = dict(
            include_previous=True,
            truncate=args.truncate_data,
            include_partial=not args.truncate_data
        )
    train_data = (
        load_data_file(args.train_data, vocab, args.batch_size, args.bptt_limit, device, **kwargs)
        if train else None)
    valid_data = (
        load_data_file(args.valid_data, vocab, args.eval_batch_size, args.eval_bptt_limit, device, **kwargs)
        if valid else None)
    test_data = (
        load_data_file(args.test_data, vocab, args.eval_batch_size, args.eval_bptt_limit, device, **kwargs)
        if test else None)
    return Data(train_data, valid_data, test_data, vocab)

def load_one_long_sequence(path, vocab):
    one_long_sequence = []
    with path.open() as fin:
        for line in fin:
            one_long_sequence.extend(vocab.as_index(s) for s in line.split())
            # Make sure to add <eos> after each sentence.
            one_long_sequence.append(vocab.eos)
    return one_long_sequence

def load_data_file(path, vocab, batch_size, bptt_limit, device,
        include_previous=True, truncate=False, include_partial=True):
    one_long_sequence = load_one_long_sequence(path, vocab)
    if truncate:
        # Optionally truncate the dataset so it is evenly divisible by the
        # batch size and all batches have the same batch size.
        divisible_length = (len(one_long_sequence) // batch_size) * batch_size
        del one_long_sequence[divisible_length:]
    # Wrap the entire dataset around into `batch_size` batch elements. Figure
    # out how long each batch element will be by dividing the total length by
    # `batch_size` and rounding up.
    batch_element_length = math.ceil(len(one_long_sequence) / batch_size)
    # Split the dataset into batch elements of length
    # `batch_element_length + 1`. The last symbol in each batch element is
    # copied to the first symbol of the next, hence the +1. The first symbol
    # in each batch element is only used as an input and not counted in
    # evaluation or the training objective; every symbol in the training set
    # is counted exactly once. The first symbol in the first batch element is
    # <eos>, which is an appropriate choice because it marks sentence
    # boundaries anyway.
    batch_elements = []
    prev_symbol = vocab.eos
    for chunk in more_itertools.chunked(one_long_sequence, batch_element_length):
        batch_element = []
        if include_previous:
            # Optionally include the last symbol of the previous batch element.
            batch_element.append(prev_symbol)
        batch_element.extend(chunk)
        prev_symbol = batch_element[-1]
        batch_elements.append(batch_element)
    # Pad the last batch element with <eos> (these will be ignored anyway).
    # TODO It might be better to construct two big tensors, one before and one
    # after the split point.
    last_batch_element = batch_elements[-1]
    last_batch_element_length = len(last_batch_element)
    padding_length = batch_element_length + int(include_previous) - last_batch_element_length
    last_batch_element.extend(vocab.eos for i in range(padding_length))
    # Convert the data to one big tensor.
    one_big_tensor = torch.tensor(batch_elements, device=device)
    # Split the tensor into chunks of length `bptt_limit`.
    tensor_pairs = []
    offset = 0
    while offset < batch_element_length:
        # The last batch element will probably be jagged -- it will contain
        # padding symbols that should be ignored. When we split the big tensor
        # into chunks, this will result in at most one chunk that will be
        # jagged. We will make it un-jagged by splitting it into two chunks at
        # the point where the padding in the last batch element starts. Chunks
        # that occur before the split point will have batch size `batch_size`,
        # and chunks that occur after the split point will have batch size
        # `batch_size - 1`.
        actual_chunk_length = min(batch_element_length - offset, bptt_limit)
        if not include_partial and actual_chunk_length < bptt_limit:
            # If the chunk is less than the requested BPTT length, optionally
            # exclude it.
            break
        num_jagged = last_batch_element_length - offset
        if num_jagged <= 0:
            # There are no symbols in the last batch element, so we are after
            # the split point.
            x = one_big_tensor[:-1, offset:offset+actual_chunk_length]
            y = one_big_tensor[:-1, offset+1:offset+1+actual_chunk_length]
            tensor_pairs.append((x, y))
        elif num_jagged >= actual_chunk_length:
            # The last batch element is full of symbols, so we are before the
            # split point.
            x = one_big_tensor[:, offset:offset+actual_chunk_length]
            y = one_big_tensor[:, offset+1:offset+1+actual_chunk_length]
            tensor_pairs.append((x, y))
        else:
            # The last batch element is partially full of symbols. Split the
            # chunk in the middle.
            # 0 < num_jagged < actual_chunk_length
            x = one_big_tensor[:, offset:offset+num_jagged]
            y = one_big_tensor[:, offset+1:offset+1+num_jagged]
            tensor_pairs.append((x, y))
            x = one_big_tensor[:-1, offset+num_jagged:offset+actual_chunk_length]
            y = one_big_tensor[:-1, offset+1+num_jagged:offset+1+actual_chunk_length]
            tensor_pairs.append((x, y))
        offset += bptt_limit
    return tensor_pairs
