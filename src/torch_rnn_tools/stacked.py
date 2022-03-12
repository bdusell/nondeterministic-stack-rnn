import typing

import torch

from .unidirectional_rnn import UnidirectionalRNN

class StackedRNN(UnidirectionalRNN):
    """Stacks one RNN on another."""

    def __init__(self, first: UnidirectionalRNN, second: UnidirectionalRNN):
        """The output of the first RNN will be fed into the input of the second
        RNN. The output size of the first must match the input size of the
        second."""
        super().__init__()
        self.first = first
        self.second = second
        if first.output_size() != second.input_size():
            raise ValueError(
                f'the output size of the first RNN ({first.output_size()}) '
                f'must match the input size of the second RNN ({second.input_size()})')

    def input_size(self):
        return self.first.input_size()

    def output_size(self):
        return self.second.output_size()

    class State(UnidirectionalRNN.State):

        def __init__(self, first_state, second_state):
            self.first_state = first_state
            self.second_state = second_state

        def next(self, input_tensor):
            new_first_state = self.first_state.next(input_tensor)
            new_second_state = self.second_state.next(new_first_state.output())
            return UnidirectionalStackedRNN.State(new_first_state, new_second_state)

        def output(self):
            return self.second_state.output()

        def detach(self):
            return StackedRNN.State(self.first_state.detach(), self.second_state.detach())

        def batch_size(self):
            return self.first_state.batch_size()

        def slice_batch(self, s):
            return StackedRNN.State(self.first_state.slice_batch(s), self.second_state.slice_batch(s))

        # TODO override fastforward, states, and outputs

        def forward(self, input_sequence, return_state, include_first):
            first_kwargs = dict(
                return_state=return_state,
                include_first=False
            )
            second_kwargs = dict(
                return_state=return_state,
                include_first=include_first
            )
            if return_state:
                first_output, first_state = \
                    self.first_state.forward(input_sequence, **first_kwargs)
                second_output, second_state = \
                    self.second_state.forward(first_output, **second_kwargs)
                state = self.State(first_state, second_state)
                return second_output, state
            else:
                first_output = \
                    self.first_state.forward(input_sequence, **first_kwargs)
                second_output = \
                    self.second_state.forward(first_output, **second_kwargs)
                return second_output

    def initial_state(self, batch_size):
        return self.State(
            self.first.initial_state(batch_size),
            self.second.initial_state(batch_size)
        )
