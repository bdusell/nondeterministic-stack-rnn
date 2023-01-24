import torch_semiring_einsum

from torch_extras.init import init_modules_by_type
from torch_rnn_tools import with_embeddings, UnidirectionalLSTM
from stack_rnn_models.joulin_mikolov import JoulinMikolovRNN
from stack_rnn_models.limited_nondeterministic_stack import LimitedNondeterministicStackRNN
from stack_rnn_models.limited_vector_nondeterministic_stack import LimitedVectorNondeterministicStackRNN
from lib.pytorch_tools.model_interface import ModelInterface
from cfl_language_modeling.model_util import print_example_outputs
from .semeniuta_lstm import SemeniutaLSTM

class NaturalModelWithContextInterface(ModelInterface):

    def add_more_init_arguments(self, group):
        group.add_argument('--model-type', choices=[
            'lstm',
            'semeniuta-lstm'
        ], default='lstm',
            help='The type of model to use.')
        group.add_argument('--hidden-units', type=int,
            default=20,
            help='The number of hidden units per layer in the LSTM.')
        group.add_argument('--layers', type=int,
            default=1,
            help='The number of layers used in the LSTM.')
        group.add_argument('--use-xavier-init', action='store_true', default=False,
            help='Use Xavier initialization for all fully-connected layers '
                 '(only applies to instances of the Layer class). All other '
                 'parameters are randomly initialized from a uniform '
                 'distribution.')
        group.add_argument('--uniform-init-scale', type=float, default=0.1,
            help='Scales the interval from which parameters are initialized '
                 'uniformly where uniform initialization is used.')
        group.add_argument('--input-dropout', type=float, default=None,
            help='Dropout applied to the inputs to the LSTM. Default is no '
                 'dropout.')
        group.add_argument('--layer-dropout', type=float, default=None,
            help='Dropout applied in between LSTM layers. Default is no '
                 'dropout. This is ignored if there is only one layer.')
        group.add_argument('--output-dropout', type=float, default=None,
            help='Dropout applied to the outputs of the LSTM. Default is no '
                 'dropout.')
        group.add_argument('--recurrent-dropout', type=float, default=None,
            help='(semeniuta-lstm) Dropout applied to the candidate hidden '
                 'state.')
        group.add_argument('--tied-embeddings', action='store_true', default=False,
            help='Tie the input and output word embeddings.')
        group.add_argument('--stack-model', choices=[
            'none',
            'jm',
            'ns',
            'vns'
        ], default='none',
            help='Type of stack data structure to attach to the model.')
        group.add_argument('--jm-push-type',
            help='(jm) Whether to push hidden states or learned vectors onto '
                 'the stack.',
            choices=[
                'hidden-state',
                'learned'
            ], default='learned')
        group.add_argument('--stack-embedding-size', type=int,
            help='(jm, vns) The size of the vectors used in the stack.')
        group.add_argument('--stack-depth-limit', type=int,
            help='(jm) The limit of the depth of the stack.')
        group.add_argument('--num-states', type=int,
            help='(ns, vns) Number of PDA states.')
        group.add_argument('--stack-alphabet-size', type=int,
            help='(ns, vns) Number of symbol types in the stack alphabet.')
        group.add_argument('--window-size', type=int,
            help='(ns, vns) Size of the window in the limited NS-RNN stack.')
        group.add_argument('--normalize-operations', action='store_true', default=False,
            help='(ns, vns) Normalize the operation weights.')
        group.add_argument('--no-states-in-reading', action='store_true', default=False,
            help='(ns) Do not include PDA states in the stack reading.')

    def add_forward_arguments(self, parser):
        group = parser.add_argument_group('Forward pass options')
        group.add_argument('--block-size', type=int,
            default=torch_semiring_einsum.AutomaticBlockSize(),
            help='(ns, vns) The block size used in einsum operations. Default is automatic.')

    def save_args(self, args):
        self.block_size = args.block_size

    def get_kwargs(self, args, vocab):
        return dict(
            model_type=args.model_type,
            vocabulary_size=vocab.size(),
            hidden_units=args.hidden_units,
            layers=args.layers,
            input_dropout=args.input_dropout,
            layer_dropout=args.layer_dropout,
            output_dropout=args.output_dropout,
            recurrent_dropout=args.recurrent_dropout,
            tied_embeddings=args.tied_embeddings,
            stack_model=args.stack_model,
            jm_push_type=args.jm_push_type,
            stack_embedding_size=args.stack_embedding_size,
            stack_depth_limit=args.stack_depth_limit,
            num_states=args.num_states,
            stack_alphabet_size=args.stack_alphabet_size,
            window_size=args.window_size,
            normalize_operations=args.normalize_operations,
            include_states_in_reading=not args.no_states_in_reading
        )

    def construct_model(self, model_type, vocabulary_size, hidden_units,
            layers, input_dropout, layer_dropout, output_dropout,
            recurrent_dropout, tied_embeddings, stack_model, jm_push_type,
            stack_embedding_size, stack_depth_limit, num_states,
            stack_alphabet_size, window_size, normalize_operations,
            include_states_in_reading):
    
        if model_type == 'lstm':
            if recurrent_dropout is not None:
                raise ValueError('lstm does not support recurrent dropout')
            def make_rnn(input_size):
                return UnidirectionalLSTM(
                    input_size=input_size,
                    hidden_units=hidden_units,
                    layers=layers,
                    dropout=layer_dropout
                )
        elif model_type == 'semeniuta-lstm':
            if layers is not None and layers != 1:
                raise ValueError('multiple layers not supported for semeniuta-lstm')
            def make_rnn(input_size):
                return SemeniutaLSTM(
                    input_size=input_size,
                    hidden_units=hidden_units,
                    dropout=recurrent_dropout
                )
        else:
            raise ValueError

        embedding_size = hidden_units

        if stack_model == 'none':
            rnn = make_rnn(embedding_size)
        elif stack_model == 'jm':
            push_hidden_state = jm_push_type == 'hidden-state'
            if push_hidden_state:
                if stack_embedding_size is not None:
                    raise ValueError('do not use --stack-embedding-size with --jm-push-type hidden-state')
                stack_embedding_size = hidden_units
            else:
                if stack_embedding_size is None:
                    raise ValueError('--stack-embedding-size is missing')
            if stack_depth_limit is None:
                raise ValueError('--stack-depth-limit is missing')
            rnn = JoulinMikolovRNN(
                input_size=embedding_size,
                stack_embedding_size=stack_embedding_size,
                controller=make_rnn,
                push_hidden_state=push_hidden_state,
                stack_depth_limit=stack_depth_limit
            )
        elif stack_model == 'ns':
            if num_states is None:
                raise ValueError('--num-states is missing')
            if stack_alphabet_size is None:
                raise ValueError('--stack-alphabet-size is missing')
            if window_size is None:
                raise ValueError('--window-size')
            rnn = LimitedNondeterministicStackRNN(
                input_size=embedding_size,
                num_states=num_states,
                stack_alphabet_size=stack_alphabet_size,
                window_size=window_size,
                controller=make_rnn,
                normalize_operations=normalize_operations,
                include_states_in_reading=include_states_in_reading
            )
        elif stack_model == 'vns':
            if stack_embedding_size is None:
                raise ValueError('--stack-embedding-size is missing')
            if num_states is None:
                raise ValueError('--num-states is missing')
            if stack_alphabet_size is None:
                raise ValueError('--stack-alphabet-size is missing')
            if window_size is None:
                raise ValueError('--window-size is missing')
            rnn = LimitedVectorNondeterministicStackRNN(
                input_size=embedding_size,
                num_states=num_states,
                stack_alphabet_size=stack_alphabet_size,
                stack_embedding_size=stack_embedding_size,
                window_size=window_size,
                controller=make_rnn,
                normalize_operations=normalize_operations
            )
        else:
            raise ValueError

        return wrap_rnn(
            rnn,
            vocabulary_size=vocabulary_size,
            input_dropout=input_dropout,
            output_dropout=output_dropout,
            tied_embeddings=tied_embeddings
        )

    def initialize(self, args, model, generator):
        def uniform_init(name, data, generator):
            data.uniform_(-args.uniform_init_scale, args.uniform_init_scale, generator=generator)
        def xavier_init(name, module, generator):
            module.xavier_uniform_init(generator=generator)
        callbacks = []
        if args.use_xavier_init:
            callbacks.append((Layer, xavier_init))
        init_modules_by_type(model, callbacks, uniform_init, generator)

    def get_initial_state(self, model, batch_size):
        return model.wrapped_rnn().initial_state(batch_size)

    def get_logits_and_state(self, saver, model_state, x, train_state):
        stack_model = saver.kwargs['stack_model']
        forward_kwargs = {}
        if stack_model in ('ns', 'vns'):
            forward_kwargs['block_size'] = self.block_size
        logits, new_model_state = saver.model(
            x,
            initial_state=model_state,
            return_state=True,
            include_first=False,
            **forward_kwargs
        )
        return logits, new_model_state

    def print_example(self, saver, batch, vocab, logger):
        def get_logits(model, x):
            logits, _ = self.get_logits_and_state(saver, None, x, None)
            return logits
        print_example_outputs(saver.model, get_logits, batch, vocab, logger)

    def get_parameter_groups(self, model):
        return model.parameters()

def wrap_rnn(rnn, vocabulary_size, input_dropout, output_dropout,
        tied_embeddings):
    return with_embeddings(
        rnn,
        vocabulary_size=vocabulary_size,
        tied=tied_embeddings,
        output_size=vocabulary_size,
        bias=not tied_embeddings,
        input_dropout=input_dropout,
        output_dropout=output_dropout
    )
