import torch

from torch_extras.init import xavier_uniform_init
from torch_rnn_tools import UnidirectionalLSTM, OutputLayerWrapper
from nsrnn.models.grefenstette import GrefenstetteRNN
from nsrnn.models.joulin_mikolov import JoulinMikolovRNN
from nsrnn.models.nondeterministic_stack import NondeterministicStackRNN
from nsrnn.pretty_table import align, green, red
from nsrnn.pytorch_tools.model_interface import ModelInterface

class CFLModelInterface(ModelInterface):

    def add_more_init_arguments(self, group):
        group.add_argument('--model-type', choices=['lstm', 'gref', 'jm', 'ns'], required=True,
            help='The type of model to use. Choices are "lstm" (LSTM), "gref" '
                 '(Grefenstette et al. 2015), "jm" (Joulin & Mikolov 2015), '
                 'and "ns" (Nondeterministic Stack RNN).')
        group.add_argument('--hidden-units', type=int,
            default=20,
            help='The number of hidden units used in the LSTM controller.')
        group.add_argument('--layers', type=int,
            default=1,
            help='The number of layers used in the LSTM controller.')
        group.add_argument('--stack-embedding-size', type=int,
            help='(gref and jm only) The size of the embeddings used in the '
                 'differentiable stack.')
        group.add_argument('--num-states', type=int,
            help='(ns only) The number of PDA states used by the NS-RNN.')
        group.add_argument('--stack-alphabet-size', type=int,
            help='(ns only) The number of symbols in the stack alphabet used '
                 'by the NS-RNN.')
        group.add_argument('--normalize-operations', action='store_true', default=False,
            help='(ns only) Normalize the stack operation weights so that '
                 'they sum to one.')
        group.add_argument('--no-states-in-reading', action='store_true', default=False,
            help='(ns only) Do not include PDA states in the stack reading.')
        group.add_argument('--init-scale', type=float, default=0.1,
            help='Scales the interval from which parameters are initialized '
                 'uniformly (fully-connected layers outside the LSTM ignore '
                 'this and always use Xavier initialization).')

    def add_forward_arguments(self, group):
        group.add_argument('--block-size', type=int, default=None,
            help='(ns only) The block size used in einsum operations.')

    def save_args(self, args):
        self.forward_kwargs = {}
        if args.block_size is not None:
            self.forward_kwargs['block_size'] = args.block_size

    def get_kwargs(self, args, input_size, output_size):
        return dict(
            model_type=args.model_type,
            input_size=input_size,
            hidden_units=args.hidden_units,
            layers=args.layers,
            output_size=output_size,
            stack_embedding_size=args.stack_embedding_size,
            num_states=args.num_states,
            stack_alphabet_size=args.stack_alphabet_size,
            normalize_operations=args.normalize_operations,
            include_states_in_reading=not args.no_states_in_reading
        )

    def construct_model(self, model_type, input_size, hidden_units,
            output_size, layers, stack_embedding_size, num_states,
            stack_alphabet_size, normalize_operations,
            include_states_in_reading=True):

        def construct_controller(input_size):
            return UnidirectionalLSTM(
                input_size=input_size,
                hidden_units=hidden_units,
                layers=layers
            )

        if model_type == 'lstm':
            rnn = construct_controller(input_size)
        elif model_type == 'gref':
            if stack_embedding_size is None:
                raise ValueError('--stack-embedding-size are required')
            rnn = GrefenstetteRNN(
                input_size=input_size,
                stack_embedding_size=stack_embedding_size,
                controller=construct_controller
            )
        elif model_type == 'jm':
            if stack_embedding_size is None:
                raise ValueError('--stack-embedding-size is required')
            rnn = JoulinMikolovRNN(
                input_size=input_size,
                stack_embedding_size=stack_embedding_size,
                controller=construct_controller,
                push_hidden_state=False
            )
        elif model_type == 'ns':
            if num_states is None or stack_alphabet_size is None:
                raise ValueError('--num-states and --stack-alphabet-size are required')
            rnn = NondeterministicStackRNN(
                input_size=input_size,
                num_states=num_states,
                stack_alphabet_size=stack_alphabet_size,
                controller=construct_controller,
                normalize_operations=normalize_operations,
                normalize_reading=True,
                include_states_in_reading=include_states_in_reading
            )
        else:
            raise ValueError

        return OutputLayerWrapper(rnn, output_size)

    def initialize(self, args, model, generator):
        def fallback(name, param, generator):
            param.data.uniform_(
                -args.init_scale,
                args.init_scale,
                generator=generator)
        xavier_uniform_init(model, generator, fallback)

    def get_logits(self, saver, x, train_state):
        return saver.model(x, **self.forward_kwargs)

    def get_signals(self, saver, x, train_data):
        model_type = saver.kwargs['model_type']
        more_kwargs = {}
        if model_type in ('gref', 'jm', 'ns'):
            logits, signals = saver.model(x, return_signals=True, include_first=False, **self.forward_kwargs)
            if model_type == 'ns':
                return [[x.tolist() for x in signals_t if x is not None] for signals_t in signals]
            else:
                raise NotImplementedError
        else:
            return None

    def print_example(self, saver, batch, vocab, logger):
        def get_logits(model, x):
            return self.get_logits(saver, x, None)
        print_example_outputs(saver.model, get_logits, batch, vocab, logger)

    def get_parameter_groups(self, model):
        return model.parameters()

def print_example_outputs(model, get_logits, batch, vocab, logger):
    x, y_target = batch
    model.eval()
    with torch.no_grad():
        # logits : B x n x V
        y_logits = get_logits(model, x)
        # y_pred : B x n
        y_pred = torch.argmax(y_logits, dim=2)
    for y_target_elem, y_pred_elem in zip(y_target.tolist(), y_pred.tolist()):
        align([
            [
                mark_color(vocab.value(p), t == p)
                for t, p
                in zip(y_target_elem, y_pred_elem)
            ],
            [vocab.value(s) for s in y_target_elem]
        ], print=logger.info)
        logger.info('')

def mark_color(s, p):
    if p:
        return green(s)
    else:
        return red(s)
