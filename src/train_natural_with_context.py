import argparse
import logging
import sys

from utils.natural_data_with_context_util import (
    add_natural_data_arguments, load_data)
from utils.model_with_context_util import NaturalModelWithContextInterface
from utils.train_with_context_util import add_train_arguments, train

def main():

    argv = sys.argv[1:]
    logger = logging.getLogger('main')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    logger.info(f'arguments: {argv}')
    model_interface = NaturalModelWithContextInterface(require_output=False)

    parser = argparse.ArgumentParser(
        description=
        'Initialize a model and train it to convergence on a natural '
        'language data set for language modeling. The model can be any of '
        'those described in the paper.'
    )
    add_natural_data_arguments(parser, train=True, valid=True)
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
    logger.info(f'device: {device}')

    data = load_data(device, args, train=True, valid=True)
    print(f'train batches: {len(data.train_data)}')
    print(f'valid batches: {len(data.valid_data)}')

    saver = model_interface.construct_saver(args, data.vocab)

    with saver.logger() as events:
        events.log('start', {
            'learning_rate' : args.learning_rate
        })
        # Train the model.
        train(
            args=args,
            saver=saver,
            parameter_groups=model_interface.get_parameter_groups(saver.model),
            train_data=data.train_data,
            valid_data=data.valid_data,
            vocab=data.vocab,
            model_interface=model_interface,
            events=events,
            logger=logger
        )

if __name__ == '__main__':
    main()
