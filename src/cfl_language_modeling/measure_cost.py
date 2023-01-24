import argparse
import datetime
import random

import attr
import humanize
import numpy
import torch

from cfl_language_modeling.model_util import CFLModelInterface
from cfl_language_modeling.data_util import indexes_to_one_hot_input_tensor

@attr.s
class Cost:
    duration = attr.ib()
    memory = attr.ib()

def main():

    model_interface = CFLModelInterface(require_output=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--vocab-size', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--average-steps', type=int, default=10)
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    args = parser.parse_args()

    model_interface.save_args(args)

    device = model_interface.get_device(args)
    Y = torch.randint(args.vocab_size, (args.batch_size, args.length), device=device)
    Y[:, -1] = args.vocab_size
    X = indexes_to_one_hot_input_tensor(Y, args.vocab_size)

    saver = model_interface.construct_saver(
        args,
        input_size=args.vocab_size,
        output_size=args.vocab_size
    )
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    torch.cuda.synchronize(device)
    base_memory = torch.cuda.memory_allocated(device)

    def measure_cost(func):
        torch.cuda.synchronize(device)
        torch.cuda.reset_max_memory_allocated(device)
        start_time = datetime.datetime.now()
        func()
        torch.cuda.synchronize(device)
        duration = datetime.datetime.now() - start_time
        memory = torch.cuda.max_memory_allocated(device) - base_memory
        return Cost(duration, memory)

    def callback():
        saver.model.train()
        logits = model_interface.get_logits(saver, X, None)
        symbol_losses = criterion(logits.transpose(1, 2), Y)
        sequence_losses = torch.sum(symbol_losses, dim=1)
        loss = torch.mean(sequence_losses, dim=0)
        loss.backward()

    # Warm up CUDA a few times.
    for i in range(args.warmup_steps):
        measure_cost(callback)

    # Get measurements for multiple iterations to average together.
    costs = [measure_cost(callback) for i in range(args.average_steps)]

    mean_duration = numpy.mean([cost.duration for cost in costs])
    memory = costs[0].memory

    print(f'duration: {humanize.precisedelta(mean_duration, minimum_unit="milliseconds")}')
    print(f'memory: {humanize.naturalsize(memory)}')

if __name__ == '__main__':
    main()
