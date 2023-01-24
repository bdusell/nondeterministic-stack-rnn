import typing

import torch

from .unidirectional_rnn import UnidirectionalRNN

class BuiltinUnidirectionalRNN(UnidirectionalRNN):
    """Wraps a built-in PyTorch RNN class in the :py:class:`UnidirectionalRNN`
    API."""

    def __init__(self, input_size: int, hidden_units: int, layers: int=1,
            dropout: typing.Optional[float]=None, **kwargs):
        """
        :param input_size: The size of the input vectors to the RNN.
        :param hidden_units: The number of hidden units in each layer.
        :param layers: The number of layers in the RNN.
        :param dropout: The amount of dropout applied in between layers. If
            ``layers`` is 1, then this value is ignored.
        :param kwargs: Additional arguments passed to the RNN constructor,
            such as ``nonlinearity`` and ``bias``.
        """
        if dropout is None or layers == 1:
            dropout = 0.0
        super().__init__()
        self.rnn = self.RNN_CLASS(
            input_size,
            hidden_units,
            num_layers=layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
            **kwargs)
        self._input_size = input_size
        self._hidden_units = hidden_units
        self._layers = layers

    def _initial_tensors(self, batch_size, first_layer):
        raise NotImplementedError

    def _apply_to_hidden_state(self, hidden_state, func):
        raise NotImplementedError

    def input_size(self):
        return self._input_size

    def output_size(self):
        return self._hidden_units

    class State(UnidirectionalRNN.State):

        def __init__(self, rnn, hidden_state, output):
            self.rnn = rnn
            self.hidden_state = hidden_state
            self._output = output

        def next(self, input_tensor):
            # input_tensor : batch_size x input_size
            # unsqueezed_input : batch_size x 1 x input_size
            unsqueezed_input = input_tensor.unsqueeze(1)
            unsqueezed_output, new_hidden_state = self.rnn.rnn(
                unsqueezed_input,
                self.hidden_state)
            # unsqueezed_output : batch_size x 1 x hidden_units
            return self.rnn.State(
                self.rnn,
                new_hidden_state,
                unsqueezed_output.squeeze(1))

        def output(self):
            return self._output

        def batch_size(self):
            return self._output.size(0)

        def slice_batch(self, s):
            return self.transform_tensors(lambda x: x[..., s, :].contiguous())

        def transform_tensors(self, func):
            return self.rnn.State(
                self.rnn,
                self.rnn._apply_to_hidden_state(self.hidden_state, func),
                func(self._output))

        def fastforward(self, input_sequence):
            """This method is overridden to use the builtin RNN class
            efficiently."""
            input_tensors = input_sequence.transpose(0, 1)
            output_sequence, state = self.forward(
                input_tensors,
                return_state=True,
                include_first=False)
            return state

        def outputs(self, input_sequence, include_first):
            """This method is overridden to use the builtin RNN class
            efficiently."""
            return self.forward(
                input_sequence,
                return_state=False,
                include_first=include_first)

        def forward(self, input_sequence, return_state, include_first):
            """This method is overridden to use the builtin RNN class
            efficiently."""
            # input_sequence : batch_size x sequence_length x input_size
            # self.output() : batch_size x hidden_units
            # Handle empty sequences, since the built-in RNN module does not
            # handle empty sequences (I checked).
            if input_sequence.size(1) == 0:
                first_output = self.output()
                if include_first:
                    output_sequence = first_output[:, None, :]
                else:
                    batch_size, hidden_units = first_output.size()
                    output_sequence = first_output.new_empty(batch_size, 0, hidden_units)
                if return_state:
                    return output_sequence, self
                else:
                    return output_sequence
            # output_sequence : batch_size x sequence_length x hidden_units
            # The type and size of new_hidden_state depends on the RNN unit.
            # For torch.nn.RNN, it's a tensor whose size is
            # num_layers x batch_size x hidden_units, where
            # new_hidden_state[-1] is the last layer and is equal to
            # output_sequence[:, -1].
            output_sequence, new_hidden_state = self.rnn.rnn(
                input_sequence,
                self.hidden_state)
            # output_sequence : batch_size x sequence_length x hidden_units
            if include_first:
                first_output = self.output()
                output_sequence = torch.cat([
                    first_output[:, None, :],
                    output_sequence
                ], dim=1)
            if return_state:
                # last_output : batch_size x hidden_units
                last_output = output_sequence[:, -1, :]
                state = self.rnn.State(self.rnn, new_hidden_state, last_output)
                return output_sequence, state
            else:
                return output_sequence

    def initial_state(self, batch_size: int, first_layer: typing.Optional[torch.Tensor]=None):
        """
        :param first_layer: An optional tensor that will be used as the first
            layer of the initial hidden state.
        """
        hidden_state, output = self._initial_tensors(batch_size, first_layer)
        return self.State(self, hidden_state, output)

class UnidirectionalElmanRNN(BuiltinUnidirectionalRNN):
    """A simple Elman RNN wrapped in the :py:class:`UnidirectionalRNN` API."""

    def __init__(self, input_size: int, hidden_units: int, layers: int=1,
            dropout: typing.Optional[float]=None,
            nonlinearity: typing.Literal['tanh', 'relu']='tanh',
            bias: bool=True):
        """
        :param input_size: The size of the input vectors to the RNN.
        :param hidden_units: The number of hidden units in each layer.
        :param layers: The number of layers in the RNN.
        :param dropout: The amount of dropout applied in between layers. If
            ``layers`` is 1, then this value is ignored.
        :param nonlinearity: The non-linearity applied to hidden units. Either
            ``'tanh'`` or ``'relu'``.
        :param bias: Whether to use bias terms.
        """
        super().__init__(
            input_size=input_size,
            hidden_units=hidden_units,
            layers=layers,
            dropout=dropout,
            nonlinearity=nonlinearity,
            bias=bias
        )

    RNN_CLASS = torch.nn.RNN

    def _initial_tensors(self, batch_size, first_layer):
        # The initial tensor is a tensor of all the hidden states of all layers
        # before the first timestep.
        # Its size needs to be num_layers x batch_size x hidden_units, where
        # index 0 is the first layer and -1 is the last layer.
        # Note that the batch dimension is always the second dimension even
        # when batch_first=True.
        if first_layer is None:
            h = torch.zeros(
                self._layers,
                batch_size,
                self._hidden_units,
                device=self.device
            )
        else:
            expected_size = (batch_size, self._hidden_units)
            if first_layer.size() != expected_size:
                raise ValueError(
                    f'first_layer should be of size {expected_size}, but '
                    f'got {first_layer.size()}')
            h = torch.cat([
                first_layer[None],
                torch.zeros(
                    self._layers - 1,
                    batch_size,
                    self._hidden_units,
                    device=self.device
                )
            ], dim=0)
        return h, h[-1]

    def _apply_to_hidden_state(self, hidden_state, func):
        return func(hidden_state)

class UnidirectionalLSTM(BuiltinUnidirectionalRNN):
    """An LSTM wrapped in the :py:class:`UnidirectionalRNN` API."""

    def __init__(self, input_size: int, hidden_units: int, layers: int=1,
            dropout: typing.Optional[float]=None, bias: bool=True):
        """
        :param input_size: The size of the input vectors to the LSTM.
        :param hidden_units: The number of hidden units in each layer.
        :param layers: The number of layers in the LSTM.
        :param dropout: The amount of dropout applied in between layers. If
            ``layers`` is 1, then this value is ignored.
        :param bias: Whether to use bias terms.
        """
        super().__init__(
            input_size=input_size,
            hidden_units=hidden_units,
            layers=layers,
            dropout=dropout,
            bias=bias
        )

    RNN_CLASS = torch.nn.LSTM

    def _initial_tensors(self, batch_size, first_layer):
        if first_layer is None:
            h = c = torch.zeros(
                self._layers,
                batch_size,
                self._hidden_units,
                device=self.device
            )
        else:
            expected_size = (batch_size, self._hidden_units)
            if first_layer.size() != expected_size:
                raise ValueError(
                    f'first_layer should be of size {expected_size}, but '
                    f'got {first_layer.size()}')
            h = torch.cat([
                first_layer[None],
                torch.zeros(
                    self._layers - 1,
                    batch_size,
                    self._hidden_units,
                    device=self.device
                )
            ], dim=0)
            c = torch.zeros(
                self._layers,
                batch_size,
                self._hidden_units,
                device=self.device
            )
        return (h, c), h[-1]

    def _apply_to_hidden_state(self, hidden_state, func):
        return tuple(map(func, hidden_state))
