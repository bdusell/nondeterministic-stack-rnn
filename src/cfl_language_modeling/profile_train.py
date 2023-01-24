import argparse
import itertools
import pathlib
import random

import torch

from cfl_language_modeling.model_util import CFLModelInterface
from cfl_language_modeling.data_util import generate_batches_with_sizes, EndSymbolVocab
from cfl_language_modeling.tasks.common import BinaryStringSampler, UnmarkedBinaryVocab
from utils.cli_util import parse_interval
from utils.profile_torch import profile

class RandomSampler(BinaryStringSampler):

    def is_marked(self):
        return False

    def num_sections(self):
        return 1

    def binary_string_to_sample(self, w):
        return w

def main():

    model_interface = CFLModelInterface(use_output=False)

    parser = argparse.ArgumentParser()
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    parser.add_argument('--length-range', type=parse_interval,
        metavar='min:max',
        default=(40, 80))
    args = parser.parse_args()
    model_interface.save_args(args)

    device = model_interface.get_device(args)

    seed = 123
    generator = random.Random(seed)
    batch_size = 10
    sampler = RandomSampler()
    valid_lengths = list(sampler.valid_lengths(*args.length_range))
    input_vocab = UnmarkedBinaryVocab()
    output_vocab = EndSymbolVocab(input_vocab)
    batches = list(generate_batches_with_sizes(
        sampler=sampler,
        batch_sizes_and_lengths=((batch_size, length) for length in valid_lengths),
        input_vocab_size=input_vocab.size(),
        generator=generator,
        device=device
    ))
    saver = model_interface.construct_saver(
        args,
        input_size=input_vocab.size(),
        output_size=output_vocab.size()
    )
    model = saver.model

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    def run_iteration(x, y):
        model.zero_grad()
        model.train()
        logits = model_interface.get_logits(saver, x, None)
        symbol_losses = criterion(logits.transpose(1, 2), y)
        sequence_losses = torch.sum(symbol_losses, dim=0)
        loss = torch.mean(sequence_losses, dim=0)
        loss.backward()
    def func():
        for x, y in batches:
            run_iteration(x, y)

    print(profile(func, device).duration)

if __name__ == '__main__':
    main()
