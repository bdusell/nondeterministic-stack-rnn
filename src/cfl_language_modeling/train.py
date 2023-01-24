import argparse
import logging
import sys

from cfl_language_modeling.lower_bound_perplexity import compute_lower_bound_perplexity
from cfl_language_modeling.data_util import (
    add_data_arguments, generate_data, batches_to_sequences)
from cfl_language_modeling.model_util import CFLModelInterface
from cfl_language_modeling.task_util import add_task_arguments, parse_task
from cfl_language_modeling.train_util import add_train_arguments, train

def main():

    logger = logging.getLogger('main')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    logger.info(f'arguments: {sys.argv}')
    model_interface = CFLModelInterface(require_output=False)

    parser = argparse.ArgumentParser(
        description=
        'Initialize a model and train it to convergence on a task. The model '
        'and task can be any of those described in the paper.'
    )
    add_data_arguments(parser)
    add_task_arguments(parser)
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_train_arguments(parser)
    parser.add_argument('--no-progress', action='store_true', default=False,
        help='Do not print progress messages.')
    parser.add_argument('--save-model', action='store_true', default=False,
        help='Save the model parameters to the output directory so that it '
             'can be re-loaded and evaluated on a test set later. The '
             'parameters will NOT be saved if this flag is not used.')
    args = parser.parse_args()
    logger.info(f'parsed arguments: {args}')

    model_interface.save_args(args)

    device = model_interface.get_device(args)
    logger.info(f'device: {device}')

    task = parse_task(parser, args)
    sampler = task.sampler
    data = generate_data(task, device, args)
    logger.info(f'data random seed: {data.random_seed}')

    saver = model_interface.construct_saver(
        args,
        input_size=task.input_vocab.size(),
        output_size=task.output_vocab.size()
    )
    if model_interface.parameter_seed is not None:
        logger.info(f'parameter random seed: {model_interface.parameter_seed}')

    valid_lower_perp = compute_lower_bound_perplexity(
        sampler=sampler,
        num_valid_lengths=len(data.valid_lengths),
        samples=batches_to_sequences(data.valid_data)
    )
    logger.info(f'lower bound dev perplexity: {valid_lower_perp:3f}')

    with saver.logger() as events:
        events.log('start', {
            'training_examples' : args.train_data_size,
            'training_batches' : len(data.train_data),
            'valid_lower_bound_perplexity' : valid_lower_perp,
            'learning_rate' : args.learning_rate
        })
        # Train the model.
        train(
            args=args,
            saver=saver,
            parameter_groups=model_interface.get_parameter_groups(saver.model),
            data=data,
            model_interface=model_interface,
            events=events,
            logger=logger
        )

if __name__ == '__main__':
    main()
