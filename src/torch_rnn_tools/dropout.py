import typing

import torch

from .unidirectional_rnn import UnidirectionalRNN
from .wrapper import Wrapper, handle_packed_sequence

class DropoutWrapper(Wrapper):
    """Base class for RNN wrappers that apply dropout."""

    def __init__(self, rnn: UnidirectionalRNN, dropout: typing.Optional[float]):
        """
        :param rnn: An RNN to be wrapped.
        :param dropout: Dropout rate. Use ``None`` or 0 for no dropout.
        """
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

class InputDropoutWrapper(DropoutWrapper):
    """Applies dropout to the inputs of an RNN."""

    def transform_input(self, x):
        return self.apply_dropout(x)

class OutputDropoutWrapper(DropoutWrapper):
    """Applies dropout to the outputs of an RNN."""

    def transform_output(self, y):
        return self.apply_dropout(y)
