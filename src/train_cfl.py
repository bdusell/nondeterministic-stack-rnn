import argparse
import logging
import sys

from nsrnn.lang_algorithm.parsing import Parser
from nsrnn.lang_algorithm.pcfg import string_log_probability
from nsrnn.lower_bound_perplexity import compute_lower_bound_perplexity
from utils.cfl_data_util import (
    add_data_arguments, generate_data, batches_to_sequences)
from utils.model_util import CFLModelInterface
from utils.cfl_task_util import add_task_arguments, parse_task
from utils.train_util import add_train_arguments, train

def main():

    argv = sys.argv[1:]
    logger = logging.getLogger('main')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    logger.info('arguments: {}'.format(argv))
    model_interface = CFLModelInterface(require_output=False)

    parser = argparse.ArgumentParser(
        description=
        'Initialize a model and train it to convergence on a task. The model '
        'and task can be any of those described in the paper.'
    )
    group = add_data_arguments(parser)
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

    model_interface.save_args(args)

    device = model_interface.get_device(args)
    logger.info('device: {}'.format(device))

    grammar, vocab = parse_task(parser, args)
    data = generate_data(grammar, vocab, device, args)

    vocab_size = vocab.size()
    saver = model_interface.construct_saver(
        args,
        input_size=vocab_size,
        output_size=vocab_size
    )

    parser = Parser(data.grammar)
    valid_lower_perp = compute_lower_bound_perplexity(
        sampler=data.sampler,
        num_valid_lengths=len(data.valid_lengths),
        string_log_probability=lambda s: string_log_probability(parser, s),
        samples=batches_to_sequences(data.valid_data)
    )
    logger.info(f'lower bound dev perplexity: {valid_lower_perp:3f}')

    with saver.logger() as events:
        events.log('start', {
            'training_examples' : args.train_data_size,
            'training_batches' : len(data.train_data),
            'valid_lower_bound_perplexity' : valid_lower_perp,
            'init_scale' : args.init_scale,
            'learning_rate' : args.learning_rate,
            'stack_embedding_size' : args.stack_embedding_size
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
