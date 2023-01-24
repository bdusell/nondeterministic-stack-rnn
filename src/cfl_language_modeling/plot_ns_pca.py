import argparse

import sklearn.decomposition
import torch
import torch_semiring_einsum

from utils.plot_util import add_plot_arguments, run_plot
from utils.cli_util import parse_interval
from cfl_language_modeling.model_util import CFLModelInterface
from cfl_language_modeling.task_util import add_task_arguments, parse_task
from cfl_language_modeling.data_util import (
    get_random_generator,
    generate_batches,
    batches_to_sequences
)
from cfl_language_modeling.plot_ns_sample_heatmap import get_reading_vectors

def main():

    model_interface = CFLModelInterface(use_load=True, use_init=False, use_output=False)

    parser = argparse.ArgumentParser()
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_task_arguments(parser)
    parser.add_argument('--eos-in-input', action='store_true', default=False)
    parser.add_argument('--data-seed', type=int)
    parser.add_argument('--length', type=parse_interval, required=True)
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    add_plot_arguments(parser)
    args = parser.parse_args()

    device = model_interface.get_device(args)
    task = parse_task(parser, args)
    generator = get_random_generator(args.data_seed)
    batches = list(generate_batches(
        sampler=task.sampler,
        valid_lengths=task.sampler.valid_lengths(*args.length),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        input_vocab_size=task.input_vocab.size(),
        generator=generator,
        device=device,
        input_includes_eos=args.eos_in_input
    ))
    samples_as_ints = batches_to_sequences(batches, include_eos=True)
    samples_as_strs = [
        [task.output_vocab.value(x) for x in sample]
        for sample in samples_as_ints
    ]

    saver = model_interface.construct_saver(
        args,
        input_size=task.input_vocab.size(),
        output_size=task.output_vocab.size()
    )
    model = saver.model

    labels = []
    vectors = []
    with torch.no_grad():
        model.eval()
        for batch in batches:
            x, y = batch
            y_pred, readings = model(
                x,
                return_readings=True,
                block_size=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE
            )
            # readings : length x [ batch_size x stack_reading_size ]
            readings = readings[1:]
            # readings : batch_size x length x stack_reading_size
            readings = torch.stack(readings, dim=1)
            samples_as_ints = batches_to_sequences([batch], include_eos=True)
            for sample_readings, sample_ints in zip(readings, samples_as_ints):
                # sample_readings
                sample_strs = [task.output_vocab.value(s) for s in sample_ints]
                sample_labels, sample_vectors = get_reading_vectors(sample_strs, sample_readings, args.task)
                labels.extend(sample_labels)
                vectors.extend(sample_vectors)
    # vectors : num_samples x [ stack_reading_size ]
    # vectors : num_samples x stack_reading_size
    vectors = torch.stack(vectors, dim=0)
    vectors = vectors.cpu().numpy()

    with run_plot(args) as (fig, ax):
        pca = sklearn.decomposition.PCA(n_components=2)
        # pca_vectors : num_samples x 2
        pca_vectors = pca.fit_transform(vectors)
        plot = ax.scatter(pca_vectors[:, 0], pca_vectors[:, 1], c=labels)

if __name__ == '__main__':
    main()
