import torch

from nsrnn.models.grefenstette import GrefenstetteRNN
from nsrnn.models.joulin_mikolov import JoulinMikolovRNN
from nsrnn.models.nondeterministic_stack import NondeterministicStackRNN
from nsrnn.pretty_table import align, green, red
from nsrnn.pytorch_tools.init import xavier_uniform_init
from nsrnn.pytorch_tools.model_interface import ModelInterface as _ModelInterface
from nsrnn.pytorch_tools.rnn import LSTM
from nsrnn.pytorch_tools.wrappers import OutputLayerWrapper

class ModelInterface(_ModelInterface):

    def add_more_init_arguments(self, group):
        group.add_argument('--model-type', choices=['lstm', 'gref', 'jm', 'ns'], required=True,
            help='The type of model to use. Choices are "lstm" (LSTM), "gref" '
                 '(Grefenstette et al. 2015), "jm" (Joulin & Mikolov 2015), '
                 'and "ns" (Nondeterministic Stack RNN).')
        group.add_argument('--hidden-units', type=int,
            help='The number of hidden units used in the LSTM controller.')
        group.add_argument('--stack-embedding-size', type=int,
            help='(gref and jm only) The size of the embeddings used in the '
                 'differentiable stack.')
        group.add_argument('--num-states', type=int,
            help='(ns only) The number of PDA states used by the NS-RNN.')
        group.add_argument('--stack-alphabet-size', type=int,
            help='(ns only) The number of symbols in the stack alphabet used '
                 'by the NS-RNN.')
        group.add_argument('--block-size', type=int, default=10,
            help='(ns only) The block size used in einsum operations.')
        group.add_argument('--init-scale', type=float, default=0.1,
            help='Scales the interval from which parameters are initialized '
                 'uniformly (fully-connected layers outside the LSTM ignore '
                 'this and always use Xavier initialization).')
        group.add_argument('--dropout', type=float,
            help='Dropout rate used for the output layer of the LSTM '
                 'controller. Default is no dropout.')

    def save_args(self, args):
        self.block_size = args.block_size

    def get_kwargs(self, args, input_size, output_size):
        return dict(
            model_type=args.model_type,
            input_size=input_size,
            output_size=output_size,
            hidden_units=args.hidden_units,
            stack_embedding_size=args.stack_embedding_size,
            num_states=args.num_states,
            stack_alphabet_size=args.stack_alphabet_size,
            dropout=args.dropout
        )

    def construct_model(self, model_type, input_size, output_size,
            hidden_units, stack_embedding_size, num_states,
            stack_alphabet_size, dropout):
        if model_type == 'lstm':
            if hidden_units is None:
                raise ValueError('--hidden-units is required')
            rnn = LSTM(
                input_size=input_size,
                hidden_units=hidden_units
            )
        elif model_type == 'gref':
            if hidden_units is None or stack_embedding_size is None:
                raise ValueError('--hidden-units and --stack-embedding-size are required')
            rnn = GrefenstetteRNN(
                input_size=input_size,
                hidden_units=hidden_units,
                stack_embedding_size=stack_embedding_size,
                synchronized=True
            )
        elif model_type == 'jm':
            if hidden_units is None or stack_embedding_size is None:
                raise ValueError('--hidden-units and --stack-embedding-size are required')
            rnn = JoulinMikolovRNN(
                input_size=input_size,
                hidden_units=hidden_units,
                stack_embedding_size=stack_embedding_size,
                synchronized=True
            )
        elif model_type == 'ns':
            if hidden_units is None or num_states is None or stack_alphabet_size is None:
                raise ValueError(
                    '--hidden-units, --num-states, and --stack-alphabet-size are required')
            rnn = NondeterministicStackRNN(
                input_size=input_size,
                hidden_units=hidden_units,
                num_states=num_states,
                stack_alphabet_size=stack_alphabet_size
            )
        else:
            raise ValueError
        return OutputLayerWrapper(
            rnn,
            output_size=output_size,
            dropout=dropout
        )

    def initialize(self, args, model, generator):
        def fallback(name, param, generator):
            param.data.uniform_(
                -args.init_scale,
                args.init_scale,
                generator=generator)
        xavier_uniform_init(model, generator, fallback)

    def get_logits(self, model, x, train_state):
        if type(model.rnn) is NondeterministicStackRNN:
            return model(x, block_size=self.block_size)
        else:
            return model(x)

    def print_example(self, model, batch, vocab, logger):
        x, y_target = batch
        model.eval()
        with torch.no_grad():
            # logits : B x n x V
            logits = self.get_logits(model, x, None)
            # y_pred : B x n
            y_pred = torch.argmax(logits, dim=2)
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

    def get_parameter_groups(self, model):
        return model.parameters()

def mark_color(s, p):
    if p:
        return green(s)
    else:
        return red(s)
