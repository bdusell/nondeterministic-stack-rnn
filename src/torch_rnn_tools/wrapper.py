import torch

from .unidirectional_rnn import UnidirectionalRNN
from ._utils import apply_to_first_element

class Wrapper(UnidirectionalRNN):
    """Base class for wrappers that add functionality to
    :py:class:`UnidirectionalRNN`s."""

    def __init__(self, rnn: UnidirectionalRNN):
        super().__init__()
        self.rnn = rnn

    def transform_input(self, x):
        """Override this to transform inputs to the RNN. This should work both
        when the input includes a time step dimension and when it does not."""
        return x

    def transform_output(self, y):
        """Override this to transform outputs from the RNN. This should work
        both when the output includes a time step dimension and when it does
        not."""
        return y

    def forward(self, x, *args, **kwargs):
        x = apply_to_first_element(self.transform_input, x)
        y = self.rnn(x, *args, **kwargs)
        y = apply_to_first_element(self.transform_output, y)
        return y

    def input_size(self):
        return self.rnn.input_size()

    def output_size(self):
        return self.rnn.output_size()

    def wrapped_rnn(self):
        return self.rnn.wrapped_rnn()

    def wrap_input(self, x):
        return self.rnn.wrap_input(self.transform_input(x))

    class State(UnidirectionalRNN.State):

        def __init__(self, rnn, state):
            super().__init__()
            self.rnn = rnn
            self.state = state

        def next(self, input_tensor):
            return self.rnn.State(
                self.rnn,
                self.state.next(self.rnn.transform_input(input_tensor))
            )

        def output(self):
            return self.rnn.transform_output(self.state.output())

        def detach(self):
            return self.rnn.State(self.rnn, self.state.detach())

        def batch_size(self):
            return self.state.batch_size()

        def slice_batch(self, s):
            return self.rnn.State(self.rnn, self.state.slice_batch(s))

    def initial_state(self, batch_size, *args, **kwargs):
        return self.State(self, self.rnn.initial_state(batch_size, *args, **kwargs))

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
