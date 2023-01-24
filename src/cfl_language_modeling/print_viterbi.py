import argparse
import collections
import itertools
import pathlib

import torch

from cfl_language_modeling.model_util import CFLModelInterface
from cfl_language_modeling.task_util import add_task_arguments, parse_task
from cfl_language_modeling.data_util import batches_to_sequences
from cfl_language_modeling.plot_ns_sample_heatmap import read_input_string
from stack_rnn_models.nondeterministic_stack import PushOperation, ReplaceOperation, PopOperation
from lib.pretty_table import align

def get_distinct_viterbi_paths(decoder, n):
    # This is not an efficient algorithm. :(
    distinct_paths = collections.OrderedDict()
    for i in reversed(range(1, n+1)):
        (path,), scores = decoder.decode_timestep(i)
        path = tuple(path)
        is_distinct = True
        for existing_path in distinct_paths.keys():
            if is_prefix_of(path, existing_path):
                distinct_paths[existing_path] += 1
                is_distinct = False
        if is_distinct:
            distinct_paths[path] = 1
    return distinct_paths

def is_prefix_of(list1, list2):
    return len(list1) <= len(list2) and list1 == list2[:len(list1)]

def format_stack_symbol(s):
    if s == 0:
        return 'bot'
    else:
        return str(s-1)

def format_operation(op):
    if isinstance(op, PushOperation):
        return f'push {op.state_to}, {format_stack_symbol(op.symbol)}'
    elif isinstance(op, ReplaceOperation):
        return f'repl {op.state_to}, {format_stack_symbol(op.symbol)}'
    elif isinstance(op, PopOperation):
        return f'pop {op.state_to}'
    else:
        raise TypeError

def add_stack_depths(ops):
    depth = 0
    for op in ops:
        yield op, depth
        if isinstance(op, PushOperation):
            depth += 1
        elif isinstance(op, ReplaceOperation):
            pass
        elif isinstance(op, PopOperation):
            depth -= 1
        else:
            raise TypeError
        if depth < 0:
            raise ValueError

def pad_lists(rows, pad_value=''):
    max_len = max(map(len, rows))
    for row in rows:
        while len(row) < max_len:
            row.append('')

def transpose_jagged_lists(rows):
    pad_lists(rows)
    return zip(*rows)

def main():

    model_interface = CFLModelInterface(use_load=True, use_init=False, use_output=False)

    parser = argparse.ArgumentParser()
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_task_arguments(parser)
    parser.add_argument('--eos-in-input', action='store_true', default=False)
    parser.add_argument('--input-string', type=pathlib.Path, required=True)
    parser.add_argument('--horizontal', action='store_true', default=False)
    args = parser.parse_args()

    device = model_interface.get_device(args)
    task = parse_task(parser, args)
    batch_x, batch_y = batch = read_input_string(
        args.input_string,
        task=task,
        device=device,
        eos_in_input=args.eos_in_input
    )
    sample_ints, = batches_to_sequences([batch], include_eos=True)
    sample_strs = [task.output_vocab.value(x) for x in sample_ints]
    saver = model_interface.construct_saver(
        args,
        input_size=task.input_vocab.size(),
        output_size=task.output_vocab.size()
    )
    model = saver.model

    model.eval()
    with torch.no_grad():
        decoder = model.wrapped_rnn().viterbi_decoder(batch_x, args.block_size, wrapper=model)
        n = batch_x.size(1)
    paths = get_distinct_viterbi_paths(decoder, n-1)
    paths_and_counts = sorted(paths.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    rows = [[symbol] for symbol in sample_strs]
    for path, count in paths_and_counts:
        for row, op_and_depth in itertools.zip_longest(rows, itertools.chain([None], add_stack_depths(path))):
            if op_and_depth is not None:
                op, depth = op_and_depth
                row.extend('' for i in range(depth))
                row.append(format_operation(op))
        pad_lists(rows)
    if args.horizontal:
        rows = zip(*rows)
    align(rows)

if __name__ == '__main__':
    main()
