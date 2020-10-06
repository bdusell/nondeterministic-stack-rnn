import torch

from .layer import Layer, FeedForward
from .unidirectional_rnn import UnidirectionalRNNBase
from ._common import apply_to_first_element

class WrapperBase(UnidirectionalRNNBase):

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    def forward(self, x, *args, **kwargs):
        x = apply_to_first_element(self.transform_input, x)
        y = self.rnn(x, *args, **kwargs)
        y = apply_to_first_element(self.transform_output, y)
        return y

    def initial_state(self, batch_size, *args, **kwargs):
        return self.State(self, self.rnn.initial_state(batch_size, *args, **kwargs))

    def input_size(self):
        return self.rnn.input_size()

    def output_size(self):
        return self.rnn.output_size()

    def wrapped_rnn(self):
        return self.rnn.wrapped_rnn()

    def transform_input(self, x):
        return x

    def transform_output(self, y):
        return y

    def batched_next_and_output(self, states, input_tensor):
        unwrapped_states = (s.state for s in states)
        next_states, output_tensor = self.rnn.batched_next_and_output(
            unwrapped_states, self.transform_input(input_tensor))
        wrapped_next_states = (self.State(self, s) for s in next_states)
        return wrapped_next_states, self.transform_output(output_tensor)

    class State(UnidirectionalRNNBase.State):

        def __init__(self, rnn, state):
            super().__init__()
            self.rnn = rnn
            self.state = state

        def next(self, input_tensor):
            return self.rnn.State(
                self.rnn,
                self.state.next(self.rnn.transform_input(input_tensor)))

        def output(self):
            return self.rnn.transform_output(self.state.output())

class EmbeddingWrapper(WrapperBase):

    def __init__(self, rnn, vocabulary_size, sparse=False, padding_index=None,
            dropout=None, embedding_size=None):
        if dropout is not None:
            rnn = InputDropoutWrapper(rnn, dropout)
        super().__init__(rnn)
        if embedding_size is None:
            embedding_size = rnn.input_size()
        self.embedding_layer = torch.nn.Embedding(
            vocabulary_size,
            embedding_size,
            sparse=sparse,
            padding_idx=padding_index
        )

    def transform_input(self, x):
        return handle_packed_sequence(self.embedding_layer, x)

    def input_size(self):
        raise NotImplementedError

def handle_packed_sequence(func, x):
    if isinstance(x, torch.nn.utils.rnn.PackedSequence):
        return apply_to_packed_sequence(func, x)
    else:
        return func(x)

def apply_to_packed_sequence(func, x):
    return torch.nn.utils.rnn.PackedSequence(
        func(x.data),
        x.batch_sizes,
        x.sorted_indices,
        x.unsorted_indices
    )

class OutputLayerWrapper(WrapperBase):

    def __init__(self, rnn, output_size, activation=None, dropout=None):
        if dropout is not None:
            rnn = OutputDropoutWrapper(rnn, dropout)
        super().__init__(rnn)
        self.layer = Layer(rnn.output_size(), output_size, activation)

    def transform_output(self, y):
        return self.layer(y)

    def output_size(self):
        return self.layer.output_size()

class FeedForwardOutputWrapper(WrapperBase):

    def __init__(self, rnn, layer_sizes, activation):
        super().__init__(rnn)
        self.feedforward = FeedForward(
            rnn.output_size(),
            layer_sizes,
            activation
        )

    def transform_output(self, y):
        return self.feedforward(y)

    def output_size(self):
        return self.feedforward.output_size()

class DropoutWrapperBase(WrapperBase):

    def __init__(self, rnn, dropout):
        super().__init__(rnn)
        if dropout is None:
            self.dropout_layer = None
        else:
            self.dropout_layer = torch.nn.Dropout(dropout)

    def apply_dropout(self, x):
        if self.dropout_layer is None:
            return x
        else:
            return handle_packed_sequence(self.dropout_layer, x)

    def input_size(self):
        return self.rnn.input_size()

    def output_size(self):
        return self.rnn.output_size()

class InputDropoutWrapper(DropoutWrapperBase):

    def transform_input(self, x):
        return self.apply_dropout(x)

class OutputDropoutWrapper(DropoutWrapperBase):

    def transform_output(self, y):
        return self.apply_dropout(y)

class EmbeddingAndOutputLayerWrapper(WrapperBase):

    def __init__(self, rnn, vocabulary_size, output_size,
            sparse_embeddings=False, padding_index=None,
            embedding_dropout=None, output_dropout=None):
        super().__init__(
            OutputLayerWrapper(
                EmbeddingWrapper(
                    rnn,
                    vocabulary_size,
                    sparse_embeddings,
                    padding_index,
                    embedding_dropout
                ),
                output_size,
                output_dropout
            )
        )
