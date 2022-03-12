import argparse
import collections
import datetime
import json
import pathlib
import sys

from build_vocab import load_vocab, UNK_TOKEN

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True)
    parser.add_argument('--checksum', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--vocab', type=pathlib.Path, required=True)
    parser.add_argument('--dataset', choices=['ptb', 'wikitext-2'], required=True)
    args = parser.parse_args()

    curr_datetime = datetime.datetime.now(datetime.timezone.utc)
    curr_datetime_str = curr_datetime.strftime('%a %b %d %H:%M:%S %Z %Y')
    input_vocab, output_vocab = load_vocab(args.vocab)
    words = input_vocab.words
    json.dump(collections.OrderedDict([
        ('image', collections.OrderedDict([
            ('datetime', curr_datetime_str),
            ('supported_features', collections.OrderedDict([
                ('tokenize', True),
                ('unkify', True),
                ('get_surprisals', True),
                ('get_predictions', False)
            ])),
            ('gpu', collections.OrderedDict([
                ('required', False),
                ('supported', True)
            ])),
            ('maintainer', 'anonymous@example.com'),
            ('version', args.version),
            ('checksum', args.checksum)
            # size, max_memory, and max_gpu_memory appear not to be necessary.
        ])),
        ('name', args.name),
        ('ref_url', args.url),
        ('vocabulary', collections.OrderedDict([
            ('items', words + [UNK_TOKEN]),
            ('prefix_types', []),
            ('special_types', []),
            ('suffix_types', []),
            ('unk_types', [UNK_TOKEN])
        ])),
        ('tokenizer', collections.OrderedDict([
            ('cased', args.dataset == 'wikitext-2'),
            ('type', 'word')
            # TODO Drop punctuation with drop_token_pattern
        ]))
    ]), sys.stdout, indent=2)
    sys.stdout.write('\n')

if __name__ == '__main__':
    main()
