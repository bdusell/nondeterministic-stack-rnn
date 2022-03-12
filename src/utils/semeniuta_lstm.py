import torch

from torch_extras.layer import Layer
from torch_rnn_tools import UnidirectionalRNN

class SemeniutaLSTM(UnidirectionalRNN):

    def __init__(self, input_size, hidden_units, dropout):
        super().__init__()
        if dropout is None:
            dropout = 0.0
        self.recurrent_layer = Layer(
            input_size=input_size + hidden_units,
            output_size=4 * hidden_units,
            bias=True
        )
        self.dropout_layer = AlternateDropout(dropout)
        self._input_size = input_size
        self._hidden_units = hidden_units

    def input_size(self):
        return self._input_size

    def output_size(self):
        return self._hidden_units

    class State(UnidirectionalRNN.State):

        def __init__(self, rnn, hidden_state, memory_cell):
            super().__init__()
            self.rnn = rnn
            self.hidden_state = hidden_state
            self.memory_cell = memory_cell

        def next(self, input_tensor):
            layer_output = self.rnn.recurrent_layer(torch.cat([
                input_tensor,
                self.hidden_state
            ], dim=1))
            H = self.rnn._hidden_units
            gates_input, candidate_input = torch.split(layer_output, [3*H, H], dim=1)
            gates = torch.sigmoid(gates_input)
            input_gate, forget_gate, output_gate = torch.split(gates, H, dim=1)
            candidate = torch.tanh(candidate_input)
            new_memory_cell = forget_gate * self.memory_cell + input_gate * self.rnn.dropout_layer(candidate)
            new_hidden_state = output_gate * torch.tanh(new_memory_cell)
            return self.rnn.State(self.rnn, new_hidden_state, new_memory_cell)

        def output(self):
            return self.hidden_state

        def detach(self):
            return self.rnn.State(
                self.rnn,
                self.hidden_state.detach(),
                self.memory_cell.detach()
            )

        def batch_size(self):
            return self.hidden_state.size(0)

        def slice_batch(self, s):
            return self.rnn.State(
                self.rnn,
                self.hidden_state[s, ...],
                self.memory_cell[s, ...]
            )

    def initial_state(self, batch_size, initial_state=None):
        if initial_state is not None:
            if not isinstance(initial_state, self.State):
                raise TypeError
            return initial_state
        # Initial hidden state is set to 0.
        zero = torch.zeros((batch_size, self._hidden_units), device=self.device)
        return self.State(self, zero, zero)

class AlternateDropout(torch.nn.Module):
    """An implementation of dropout that uses a different scaling convention
    than PyTorch and matches the one used in the code for the paper."""

    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_layer = torch.nn.Dropout(dropout_rate)
        self.scaling_factor = 1.0 - dropout_rate

    def forward(self, x):
        return self.scaling_factor * self.dropout_layer(x)
