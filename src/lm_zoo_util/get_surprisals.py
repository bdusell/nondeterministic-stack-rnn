import argparse
import csv
import math
import pathlib
import sys

import torch

from lm_zoo_util.run_tokenize import tokenize
from utils.model_with_context_util import NaturalModelWithContextInterface
from utils.natural_data_with_context_util import add_natural_data_arguments, load_data

def main():

    model_interface = NaturalModelWithContextInterface(use_init=False, require_output=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('input_sentences', type=pathlib.Path)
    add_natural_data_arguments(parser)
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    parser.add_argument('--dataset', choices=['ptb', 'wikitext-2'], required=True)
    args = parser.parse_args()

    lowercase = args.dataset == 'ptb'

    model_interface.save_args(args)
    device = model_interface.get_device(args)
    vocab = load_data(None, args).vocab
    saver = model_interface.construct_saver(args)
    model = saver.model

    log2 = math.log(2)
    writer = csv.DictWriter(sys.stdout, [
        'sentence_id',
        'token_id',
        'token',
        'surprisal'
    ], delimiter='\t')
    writer.writeheader()

    model.eval()
    with torch.no_grad():
        with args.input_sentences.open() as fin:
            for sentence_id, line in enumerate(fin, 1):
                tokens = tokenize(line, lowercase)
                indexes = [vocab.as_index(token) for token in tokens]
                unkified_tokens = [vocab.value(index) for index in indexes]
                # x : 1 x sentence_length
                # Make sure to feed in <eos> as the first symbol, since that's
                # how the model is trained.
                x = torch.tensor([[vocab.eos] + indexes], device=device)
                # NOTE: The logits are returned in base e, but the surprisals
                # are expected in base 2.
                # logit_t = -log_e p(y_t)
                # surprisal_t = -log_2 p(y_t) = logit_t / log_2(e)
                # Remember to exclude the last symbol in x, since it doesn't
                # need to be fed as input to the model.
                # y_logits : sentence_length x vocab_size
                state = model_interface.get_initial_state(model, 1)
                (y_logits,), _ = model_interface.get_logits_and_state(model, state, x[:, :-1], None)
                y_surprisals = y_logits / log2
                for token_id, (index, token, y_surprisals_t) in \
                        enumerate(zip(indexes, unkified_tokens, y_surprisals), 1):
                    surprisal = y_surprisals_t[index].item()
                    writer.writerow({
                        'sentence_id' : sentence_id,
                        'token_id' : token_id,
                        'token' : token,
                        'surprisal' : surprisal
                    })

if __name__ == '__main__':
    main()
