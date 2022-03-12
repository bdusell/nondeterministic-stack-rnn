import argparse
import logging
import pathlib
import sys

from utils.model_with_context_util import NaturalModelWithContextInterface
from utils.train_with_context_util import evaluate
from utils.natural_data_with_context_util import add_natural_data_arguments, load_data

def main():

    argv = sys.argv[1:]
    logger = logging.getLogger('main')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    model_interface = NaturalModelWithContextInterface(use_init=False, require_output=False)

    parser = argparse.ArgumentParser()
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_natural_data_arguments(parser, test=True)
    parser.add_argument('--no-progress', action='store_true', default=False,
        help='Do not print progress messages.')
    args = parser.parse_args()

    model_interface.save_args(args)
    device = model_interface.get_device(args)
    data = load_data(device, args, test=True)
    print(f'test batches: {len(data.test_data)}')
    saver = model_interface.construct_saver(args, data.vocab)

    with saver.logger() as events:
        scores = evaluate(
            model=saver.model,
            batches=data.test_data,
            model_interface=model_interface,
            show_progress=not args.no_progress,
            logger=logger
        )
        print(f'perplexity: {scores["perplexity"]:.3f}')
        print(f'accuracy: {scores["accuracy"]:.2%}')
        events.log('test', scores)

if __name__ == '__main__':
    main()
