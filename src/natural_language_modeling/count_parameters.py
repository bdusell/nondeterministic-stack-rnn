import argparse

from natural_language_modeling.model_util import NaturalModelWithContextInterface
from natural_language_modeling.data_util import add_natural_data_arguments, load_data

def main():

    model_interface = NaturalModelWithContextInterface(
        use_load=True,
        use_output=False
    )

    parser = argparse.ArgumentParser()
    add_natural_data_arguments(parser, require_vocab=False)
    model_interface.add_arguments(parser)
    args = parser.parse_args()

    device = model_interface.get_device(args)
    data = load_data(device, args, include_vocab=args.input is None)
    saver = model_interface.construct_saver(args, data.vocab)

    num_params = sum(p.numel() for p in saver.model.parameters())
    print(num_params)

if __name__ == '__main__':
    main()
