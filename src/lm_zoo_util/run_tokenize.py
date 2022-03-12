import argparse
import pathlib

from build_vocab import load_vocab

def tokenize(line, lowercase):
    if lowercase:
        line = line.lower()
    return line.split()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pathlib.Path)
    parser.add_argument('--vocab', type=pathlib.Path, required=True)
    parser.add_argument('--unkify', action='store_true', default=False)
    parser.add_argument('--dataset', choices=['ptb', 'wikitext-2'], required=True)
    args = parser.parse_args()

    lowercase = args.dataset == 'ptb'

    input_vocab, output_vocab = load_vocab(args.vocab)
    if args.unkify:
        def unkify(s):
            return str(int(input_vocab.is_unk(s)))
    else:
        def unkify(s):
            return input_vocab.value(input_vocab.as_index(s))
    with args.input.open() as fin:
        for line in fin:
            tokens = (unkify(token) for token in tokenize(line, lowercase))
            print(' '.join(tokens))

if __name__ == '__main__':
    main()
