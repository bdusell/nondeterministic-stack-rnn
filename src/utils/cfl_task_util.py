from nsrnn.grammars.marked_reversal import (
    MarkedReversalGrammar, MarkedReversalVocab)
from nsrnn.grammars.unmarked_reversal import (
    UnmarkedReversalGrammar, UnmarkedReversalVocab)
from nsrnn.grammars.padded_reversal import (
    PaddedReversalGrammar, PaddedReversalVocab)
from nsrnn.grammars.dyck import (
    DyckGrammar, DyckVocab)
from nsrnn.grammars.hardest_cfl import (
    HardestCFLGrammar, HardestCFLVocab)
from .cli_util import get_kwargs

def add_task_arguments(parser):
    group = parser.add_argument_group('Task options')
    group.add_argument('--task',
        choices=[
            'marked-reversal',
            'unmarked-reversal',
            'padded-reversal',
            'dyck',
            'hardest-cfl'
        ],
        required=True,
        help='Choice of task. Possible choices are marked-reversal, '
             'unmarked-reversal, padded-reversal, dyck, and hardest-cfl.')
    # marked-reversal, unmarked-reversal
    group.add_argument('--symbol-types', type=int,
        default=2,
        help='(marked-reversal, unmarked-reversal) The number of symbol types '
             'in the reversed string.')
    group.add_argument('--mean-length', type=float,
        default=60,
        help='(marked-reversal, unmarked-reversal) Mean length of the '
             'reversed string.')
    # padded-reversal
    group.add_argument('--mean-content-length', type=float,
        default=60,
        help='(padded-reversal) Mean length of the reversed string.')
    group.add_argument('--mean-padding-length', type=float,
        default=30,
        help='(padded-reversal) Mean length of the padding in the middle.')
    # dyck
    group.add_argument('--bracket-types', type=int,
        default=2,
        help='(dyck) Number of bracket pair types.')
    group.add_argument('--mean-bracket-splits', type=float,
        default=1,
        help='(dyck) Mean number of bracket splits.')
    group.add_argument('--mean-nesting-depth', type=float,
        default=40,
        help='(dyck) Mean bracket nesting depth.')
    # hardest-cfl
    group.add_argument('--mean-num-commas', type=float,
        default=0.5,
        help='(hardest-cfl) Mean number of commas in each decoy string.')
    group.add_argument('--mean-short-filler-length', type=float,
        default=0.5,
        help='(hardest-cfl) Mean length of short filler strings in decoys.')
    group.add_argument('--mean-long-filler-length', type=float,
        default=2,
        help='(hardest-cfl) Mean length of long filler strings in decoys.')
    group.add_argument('--semicolon-probability', type=float,
        default=0.25,
        help='(hardest-cfl) Probability of generating a split point with a '
             'semicolon in between every symbol of the Dyck string.')

class EndSymbolVocab:

    def __init__(self, vocab, end_symbol_string='</s>'):
        super().__init__()
        self.vocab = vocab
        self.end_symbol = self.vocab.size()
        self.end_symbol_string = end_symbol_string

    def value(self, i):
        if i == self.end_symbol:
            return self.end_symbol_string
        else:
            return self.vocab.value(i)

    def size(self):
        return self.end_symbol + 1

def parse_task(parser, args):
    name = args.task
    if name == 'marked-reversal':
        grammar = MarkedReversalGrammar(**get_kwargs(parser, args, [
            'symbol_types',
            'mean_length'
        ]))
        vocab = MarkedReversalVocab(args.symbol_types)
    elif name == 'unmarked-reversal':
        grammar = UnmarkedReversalGrammar(**get_kwargs(parser, args, [
            'symbol_types',
            'mean_length'
        ]))
        vocab = UnmarkedReversalVocab(args.symbol_types)
    elif name == 'padded-reversal':
        grammar = PaddedReversalGrammar(**get_kwargs(parser, args, [
            'symbol_types',
            'mean_content_length',
            'mean_padding_length'
        ]))
        vocab = PaddedReversalVocab(args.symbol_types)
    elif name == 'dyck':
        grammar = DyckGrammar(**get_kwargs(parser, args, [
            'bracket_types',
            'mean_bracket_splits',
            'mean_nesting_depth'
        ]))
        vocab = DyckVocab(args.bracket_types)
    elif name == 'hardest-cfl':
        grammar = HardestCFLGrammar(**get_kwargs(parser, args, [
            'mean_num_commas',
            'mean_short_filler_length',
            'mean_long_filler_length',
            'semicolon_probability',
            'mean_bracket_splits',
            'mean_nesting_depth'
        ]))
        vocab = HardestCFLVocab()
    else:
        raise ValueError
    # Make sure that the vocabulary always contains a reserved end symbol.
    vocab = EndSymbolVocab(vocab)
    return grammar, vocab
