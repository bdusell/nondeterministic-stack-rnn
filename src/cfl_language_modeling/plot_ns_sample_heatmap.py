import argparse
import pathlib

import numpy
import matplotlib.pyplot as plt
import sklearn.decomposition
import torch
import torch_semiring_einsum

from cfl_language_modeling.model_util import CFLModelInterface
from cfl_language_modeling.task_util import add_task_arguments, parse_task
from cfl_language_modeling.data_util import (
    get_random_generator,
    generate_batch,
    batches_to_sequences,
    indexes_to_one_hot_input_tensor
)
from utils.plot_util import add_plot_arguments, run_plot

def read_input_string(path, task, device, eos_in_input):
    with path.open() as fin:
        tokens = fin.readline().split()
    str_to_int = {
        task.output_vocab.value(i) : i
        for i in range(task.output_vocab.size())
    }
    token_ints = [str_to_int[token] for token in tokens]
    indexes = torch.tensor([token_ints], device=device)
    x = indexes_to_one_hot_input_tensor(indexes, task.input_vocab.size() + int(eos_in_input))
    return x, indexes

def actions_to_rows(actions, plot_type):
    first_push = actions[0][0]
    Q = first_push.size(1)
    S = first_push.size(2)
    # combined_tensor : length x (Q * S) x ( push || repl || pop )
    combined_tensor = combine_action_tensors(actions)
    # normalized_tensor : length x (Q * S) x ( push || repl || pop )
    #normalized_tensor = torch.softmax(combined_tensor, dim=2)
    #normalized_tensor = torch.softmax(combined_tensor.flatten(), dim=0).view(*combined_tensor.size())
    normalized_tensor = combined_tensor
    if plot_type == 'pushes':
        normalized_tensor, _ = normalized_tensor.split([Q*S, Q*S+Q], dim=2)
    # Prepend zeros for the first timestep.
    normalized_tensor = torch.concat([
        normalized_tensor.new_zeros((1, *normalized_tensor.size()[1:])),
        normalized_tensor
    ], dim=0)
    # normalized_tensor : (Q * S) x ( push || repl || pop ) x length
    normalized_tensor = normalized_tensor.permute(1, 2, 0)
    length = normalized_tensor.size(2)
    normalized_tensor = normalized_tensor.view(-1, length)
    return normalized_tensor.tolist()

def combine_action_tensors(actions):
    # actions : length x [ (push, repl, pop) ]
    # push : 1 x Q x S x Q x S
    # repl : 1 x Q x S x Q x S
    # pop : 1 x Q x S x Q x S
    # return : length x (Q * S) x ( push || repl || pop )
    timestep_tensors = []
    for push, repl, pop in actions:
        Q = push.size(1)
        S = push.size(2)
        push = push.view(Q * S, Q * S)
        repl = repl.view(Q * S, Q * S)
        pop = pop.view(Q * S, Q)
        # timestep_tensor : (Q * S) x ( push || repl || pop )
        timestep_tensor = torch.concat([push, repl, pop], dim=1)
        timestep_tensors.append(timestep_tensor)
    return torch.stack(timestep_tensors, dim=0)

def stack_symbol_to_str(y):
    if y == 0:
        return '\\bot'
    else:
        return f'\\mathtt{{{y-1}}}'

def plot_actions(ax, rows, saver, sample_strs, plot_type):
    num_states = saver.kwargs['num_states']
    stack_alphabet_size = saver.kwargs['stack_alphabet_size']
    row_labels = []
    for q in range(num_states):
        q_str = f'q_{{{q}}}'
        for x in range(stack_alphabet_size):
            x_str = stack_symbol_to_str(x)
            push_labels = []
            repl_labels = []
            pop_labels = []
            for r in range(num_states):
                r_str = f'q_{{{r}}}'
                for y in range(stack_alphabet_size):
                    y_str = stack_symbol_to_str(y)
                    push_labels.append(f'push ${q_str}, {x_str} \\rightarrow {r_str}, {y_str}$')
                    repl_labels.append(f'repl ${q_str}, {x_str} \\rightarrow {r_str}, {y_str}$')
                pop_labels.append(f'pop ${q_str}, {x_str} \\rightarrow {r_str}$')
            row_labels.extend(push_labels)
            if plot_type != 'pushes':
                row_labels.extend(repl_labels)
                row_labels.extend(pop_labels)
    plot_heatmap(
        ax,
        rows,
        sample_strs,
        ylabel='PDA Transition',
        row_labels=row_labels
    )

def plot_readings(ax, rows, saver, sample_strs, show_y_label):
    num_states = saver.kwargs['num_states']
    stack_alphabet_size = saver.kwargs['stack_alphabet_size']
    plot_heatmap(
        ax,
        rows,
        sample_strs,
        ylabel='Reading Element',
        row_labels=[
            f'$(q_{{{r}}}, {stack_symbol_to_str(y)})$'
            for r in range(num_states)
            for y in range(stack_alphabet_size)
        ],
        show_y_label=show_y_label
    )

def edit_labels(labels):
    for label in labels:
        label = label.replace('#', '\\#')
        if label == '</s>':
            label = 'EOS'
        yield f'$\\mathtt{{{label}}}$'

def plot_heatmap(ax, rows, sample_strs, ylabel, row_labels, show_y_label):
    ax.set_xlabel(ylabel)
    if show_y_label:
        ax.set_ylabel('Input Symbol')

    data = numpy.array(rows).T

    left = 0
    top = 0
    bottom, right = data.shape
    extent = (left, right, bottom, top)

    options = dict(
        aspect='auto',
        interpolation='none',
        extent=extent
    )
    ax.imshow(data, cmap='Greys', vmin=0.0, vmax=1.0, **options)
    # Add the y axis labels in between rows.
    labels = list(edit_labels(sample_strs))
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.tick_params(axis='y', labelsize=8)
    # Remove the tick lines from the x axis.
    ax.tick_params(axis='x', length=0)
    # Add the x axis labels in the middle of columns.
    ax.set_xticks(numpy.arange(len(rows)) + 0.5, labels=row_labels, rotation=45)
    # Remove the black border.
    for k in ('top', 'right', 'bottom', 'left'):
        ax.spines[k].set_visible(False)

def get_reading_vectors(sample_strs, readings, task_name):
    # readings : length x stack_reading_size
    if task_name == 'marked-reversal':
        w_len, r = divmod(len(sample_strs)-2, 2)
        assert r == 0
        assert sample_strs[w_len] == '#'
        assert sample_strs[-1] == '</s>'
        labels = [int(s) for s in sample_strs[w_len+1:-1]]
        vectors = readings[-w_len-1:-1]
    else:
        raise NotImplementedError
    return labels, vectors

def plot_readings_pca(ax, sample_strs, readings, task_name):
    labels, vectors = get_reading_vectors(sample_strs, readings.permute(1, 0), task_name)
    vectors = vectors.permute(1, 0).cpu().numpy()
    pca = sklearn.decomposition.PCA(n_components=2)
    pca_vectors = pca.fit_transform(vectors)
    plot = ax.scatter(pca_vectors[:, 0], pca_vectors[:, 1], c=labels)
    ax.legend(handles=plot.legend_elements()[0], labels=list(map(str, labels)))

def main():

    model_interface = CFLModelInterface(use_load=True, use_init=False, use_output=False)

    parser = argparse.ArgumentParser()
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_task_arguments(parser)
    parser.add_argument('--eos-in-input', action='store_true', default=False)
    parser.add_argument('--data-seed', type=int)
    parser.add_argument('--length', type=int)
    parser.add_argument('--input-string', type=pathlib.Path)
    parser.add_argument('--show-y-label', action='store_true', default=False)
    add_plot_arguments(parser)
    parser.add_argument('--plot-type', choices=[
        'actions',
        'pushes',
        'readings',
        'readings-pca'
    ], required=True)
    args = parser.parse_args()

    device = model_interface.get_device(args)
    task = parse_task(parser, args)
    if args.input_string is not None:
        batch_x, batch_y = batch = read_input_string(
            args.input_string,
            task=task,
            device=device,
            eos_in_input=args.eos_in_input
        )
    else:
        generator = get_random_generator(args.data_seed)
        batch_x, batch_y = batch = generate_batch(
            sampler=task.sampler,
            length=args.length,
            batch_size=1,
            input_vocab_size=task.input_vocab.size(),
            generator=generator,
            device=device
        )
    sample_ints, = batches_to_sequences([batch], include_eos=True)
    sample_strs = [task.output_vocab.value(x) for x in sample_ints]
    saver = model_interface.construct_saver(
        args,
        input_size=task.input_vocab.size(),
        output_size=task.output_vocab.size()
    )
    model = saver.model

    with torch.no_grad():
        model.eval()
        batch_y_pred, actions, readings = model(
            batch_x,
            return_actions=True,
            return_readings=True,
            block_size=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE
        )

        if args.plot_type in ('actions', 'pushes'):
            actions = actions[2:]
            actions_rows = actions_to_rows(actions, args.plot_type)
            rows = actions_rows
        elif args.plot_type in ('readings', 'readings-pca'):
            readings = readings[1:]
            # readings : length x 1 x stack_reading_size
            readings = torch.stack(readings, dim=0)
            # readings : length x stack_reading_size
            readings = readings.squeeze(1)
            # readings : stack_reading_size x length
            readings = readings.permute(1, 0)
            # This is the data that will be plotted.
            readings_rows = readings.tolist()
            rows = readings_rows

    with run_plot(args) as (fig, ax):
        if args.plot_type in ('actions', 'pushes'):
            plot_actions(ax, rows, saver, sample_strs, args.plot_type)
        elif args.plot_type == 'readings':
            plot_readings(ax, rows, saver, sample_strs, args.show_y_label)
        elif args.plot_type == 'readings-pca':
            plot_readings_pca(ax, sample_strs, readings, args.task)

if __name__ == '__main__':
    main()
