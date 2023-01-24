import argparse
import pathlib

from vocab import (
    AppendedVocabulary, FallbackVocabulary, SimpleVocabulary,
    SingletonVocabulary, FixedVocabulary)
from vocab.types import unique_types

UNK_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'

def tokenize_line(line):
    return line.split()

class InputVocabulary(FallbackVocabulary):

    def __init__(self, words):
        super().__init__(
            SimpleVocabulary(words).frozen(),
            SingletonVocabulary(UNK_TOKEN)
        )

    @property
    def words(self):
        return self.main.index_to_value

    def is_unk(self, token):
        return not self.main.contains(token)

class OutputVocabulary(AppendedVocabulary):

    def __init__(self, input_vocab):
        super().__init__(
            input_vocab.frozen(),
            FixedVocabulary([EOS_TOKEN]),
            use_first=True
        )

    @property
    def words(self):
        return self.first.words

    @property
    def eos(self):
        return self.offset

    @property
    def input_slice(self):
        return [None, self.eos]

def load_vocab(path):
    with path.open() as fin:
        words = [word.rstrip() for word in fin]
    input_vocab = InputVocabulary(words)
    output_vocab = OutputVocabulary(input_vocab)
    return input_vocab, output_vocab

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()

    with args.data.open() as fin:
        words = unique_types(token for line in fin for token in tokenize_line(line) if token != UNK_TOKEN)
    input_vocab = InputVocabulary(words)
    output_vocab = OutputVocabulary(input_vocab)
    print(f'word types:        {len(output_vocab.words)}')
    print(f'input vocab size:  {input_vocab.size()}')
    print(f'output vocab size: {output_vocab.size()}')
    print(f'unk:               {output_vocab.as_index("<unk>")}')
    print(f'eos:               {output_vocab.eos}')
    with args.output.open('w') as fout:
        for word in input_vocab.words:
            print(word, file=fout)

if __name__ == '__main__':
    main()
