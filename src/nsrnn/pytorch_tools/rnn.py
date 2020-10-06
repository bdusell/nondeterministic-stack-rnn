import torch

from .unidirectional_rnn import UnidirectionalRNNBase

# Intentionally do not inherit from UnidirectionalRNNBase, since it does not
# apply if the model is bidirectional.
class RNNBase(torch.nn.Module):

    def __init__(self, input_size, hidden_units, layers=1,
            bidirectional=False, dropout=0, **kwargs):
        super().__init__()
        if dropout is None:
            dropout = 0
        RNNClass = self.rnn_class()
        self.rnn = RNNClass(input_size, hidden_units, batch_first=True,
            num_layers=layers, dropout=dropout, bidirectional=bidirectional,
            **kwargs)
        self.bidirectional = bidirectional
        self._input_size = input_size
        self._hidden_units = hidden_units
        self._layers = layers
        self._num_directions = 1 + int(bidirectional)
        self._num_init_states = self._layers * self._num_directions
        self._output_size = self._num_directions * hidden_units

    def forward(self, input_sequence, initial_state=None, return_state=False,
            include_first=True, only_last=False, return_lengths=False):
        # input_sequence : batch_size x sequence_length x input_size
        is_packed_sequence = isinstance(
            input_sequence, torch.nn.utils.rnn.PackedSequence)
        # Packed sequences cannot be empty.
        is_empty_sequence = (not is_packed_sequence) and input_sequence.size(1) == 0
        # TODO Handle empty sequences.
        if is_empty_sequence:
            raise NotImplementedError
        if is_packed_sequence:
            batch_size = input_sequence.batch_sizes[0].item()
        else:
            batch_size = input_sequence.size(0)
        # Figure out what to use for the initial state.
        if initial_state is None:
            if is_packed_sequence:
                device = input_sequence.data.device
            else:
                device = input_sequence.device
            initial_state = self.get_default_initial_state(batch_size)
        elif isinstance(initial_state, RNNBase.HiddenState):
            initial_state = initial_state.value
        # Run the built-in RNN module.
        output_tensor, final_state = self.rnn(input_sequence, initial_state)
        # output_tensor : batch_size x sequence_length x hidden_units
        lengths = None
        if only_last:
            # Only return the last hidden state.
            last_output = self.state_to_output(final_state)
            if self.bidirectional:
                last_output = torch.cat(last_output, dim=1)
            return last_output
        elif include_first:
            # The built-in RNN modules do not return the initial state in the
            # output, which means that there is no prediction for the first
            # item in a sequence. If include_first is True, prepend the initial
            # hidden state as the first output to serve as the prediction for
            # the first item in the sequence.
            if is_packed_sequence:
                # If the output is a packed sequence, pad it and make it a
                # regular tensor.
                output_tensor, lengths = torch.nn.utils.rnn.pad_packed_sequence(
                    output_tensor, batch_first=True)
            # initial_output : batch_size x hidden_units
            initial_output = self.state_to_output(initial_state)
            if self.bidirectional:
                # TODO Need to account for padding when appending the
                # backward initial state. Use the lengths returned by
                # pad_packed_sequence.
                # FIXME This is wrong when there is padding in the input.
                # initial_output_forward : batch_size x hidden_units
                # initial_output_backward : batch_size x hidden_units
                initial_output_forward, initial_output_backward = initial_output
                # output_tensor_forward : batch_size x sequence_length x hidden_units
                # output_tensor_backward : batch_size x sequence_length x hidden_units
                output_tensor_forward, output_tensor_backward = (
                    torch.chunk(output_tensor, 2, dim=2))
                output_tensor_forward = torch.cat([
                    initial_output_forward.unsqueeze(1),
                    output_tensor_forward
                ], dim=1)
                output_tensor_backward = torch.cat([
                    output_tensor_backward,
                    initial_output_backward.unsqueeze(1)
                ], dim=1)
                output_tensor = torch.cat([
                    output_tensor_forward,
                    output_tensor_backward
                ], dim=2)
            else:
                output_tensor = torch.cat([
                    initial_output.unsqueeze(1),
                    output_tensor
                ], dim=1)
        extras = []
        if return_lengths and lengths is not None:
            extras.append(lengths)
        if return_state:
            # If return_state is True and include_first is True, then do not
            # include the last hidden state in the output, but instead return
            # it separately as an extra return value. Omitting it from the
            # output tensor means that the output wrapper (the affine
            # transformation computing the output probabilities) will not
            # compute an output for it, which is what we want. The state will
            # get re-used as the initial state of a future run of the RNN, and
            # that is when it will have an output computed for it.
            if include_first:
                if is_packed_sequence:
                    raise NotImplementedError
                output_tensor = output_tensor[:, :-1]
            extras.append(self.HiddenState(final_state))
        if extras:
            return (output_tensor, *extras)
        else:
            return output_tensor

    def initial_state(self, batch_size):
        if self.bidirectional:
            raise ValueError(
                'cannot use the initial_state() API with a bidirectional RNN')
        initial_state = self.get_default_initial_state(batch_size)
        initial_output = self.state_to_hidden_state(initial_state)[-1]
        return self.State(self, initial_state, initial_output)

    def rnn_class(self):
        raise NotImplementedError

    def get_default_initial_state(self, batch_size):
        device = next(self.parameters()).device
        return self.default_initial_state(batch_size, device)

    def default_initial_state(self, batch_size, device):
        raise NotImplementedError

    def state_to_output(self, state):
        # return : batch_size x hidden_units
        #       or (batch_size x hidden_units, batch_size x hidden_units)
        # hidden_state : (layers * directions) x batch_size x hidden_units
        hidden_state = self.state_to_hidden_state(state)
        if self.bidirectional:
            return hidden_state[-2], hidden_state[-1]
        else:
            return hidden_state[-1]

    def state_to_hidden_state(self, state):
        raise NotImplementedError

    def batch_hidden_states(self, states):
        raise NotImplementedError

    def unbatch_hidden_states(self, state_batch):
        raise NotImplementedError

    def input_size(self):
        return self._input_size

    def output_size(self):
        return self._output_size

    def wrapped_rnn(self):
        return self

    def batched_next_and_output(self, states, input_tensor):
        hidden_state_batch = self.batch_hidden_states(states)
        output_tensor, next_hidden_state_batch = self.rnn(
            input_tensor.unsqueeze(1), hidden_state_batch)
        next_states = self.unbatch_hidden_states(next_hidden_state_batch)
        return next_states, output_tensor.squeeze(1)

    class HiddenState:

        def __init__(self, value):
            self.value = value

        def detach(self):
            raise NotImplementedError

    class State(UnidirectionalRNNBase.State):

        def __init__(self, rnn, hidden_state, output):
            self.rnn = rnn
            self.hidden_state = hidden_state
            self._output = output

        def next(self, input_tensor):
            # The unsqueeze() and squeeze() are there to add and remove a
            # timestep dimension, which is what the rnn() call expects.
            output, hidden_state = self.rnn.rnn(
                input_tensor.unsqueeze(1), self.hidden_state)
            return self.rnn.State(
                self.rnn, hidden_state, output.squeeze(1))

        def output(self):
            return self._output

class RNN(RNNBase):

    def rnn_class(self):
        return torch.nn.RNN

    def default_initial_state(self, batch_size, device):
        return torch.zeros(
            self._num_init_states,
            batch_size,
            self._hidden_units,
            device=device)

    def state_to_hidden_state(self, state):
        return state

    def batch_hidden_states(self, states):
        return torch.cat((s.hidden_state for s in states), dim=1)

    def unbatch_hidden_states(self, state_batch):
        for s in torch.split(state_batch, 1, dim=1):
            yield self.State(self, s, None)

    class HiddenState(RNNBase.HiddenState):

        def detach(self):
            return type(self)(self.value.detach())

class LSTM(RNNBase):

    def rnn_class(self):
        return torch.nn.LSTM

    def default_initial_state(self, batch_size, device):
        zero = torch.zeros(
            self._num_init_states,
            batch_size,
            self._hidden_units,
            device=device)
        return zero, zero

    def state_to_hidden_state(self, state):
        return state[0]

    def batch_hidden_states(self, states):
        h_list = []
        c_list = []
        for s in states:
            h, c = s.hidden_state
            h_list.append(h)
            c_list.append(c)
        h_batch = torch.cat(h_list, dim=1)
        c_batch = torch.cat(c_list, dim=1)
        return h_batch, c_batch

    def unbatch_hidden_states(self, state_batch):
        h_batch, c_batch = state_batch
        h_list = torch.split(h_batch, 1, dim=1)
        c_list = torch.split(c_batch, 1, dim=1)
        for h, c in zip(h_list, c_list):
            yield self.State(self, (h, c), None)

    class HiddenState(RNNBase.HiddenState):

        def detach(self):
            h, c = self.value
            return type(self)((h.detach(), c.detach()))

def packed_sequence_first_timestep(x):
    _require_enforce_sorted(x)
    return x.data[:x.batch_sizes[0]]

def packed_sequence_last_slice_indexes(x):
    batch_sizes = x.batch_sizes
    zero = batch_sizes.new_zeros(1)
    batch_sizes_from_zero = torch.cat([zero, batch_sizes[:-1]], dim=0)
    offsets = torch.cumsum(batch_sizes_from_zero, dim=0)
    batch_sizes_from_second = torch.cat([batch_sizes[1:], zero], dim=0)
    # TODO remove empty slices using -, eq, and mask
    lo_indexes = offsets + batch_sizes_from_second
    hi_indexes = offsets + batch_sizes
    return lo_indexes.flip(0), hi_indexes.flip(0)

def packed_sequence_last_timestep(x, indexes=None):
    _require_enforce_sorted(x)
    if indexes is None:
        indexes = packed_sequence_last_slice_indexes(x)
    lo_indexes, hi_indexes = indexes
    return torch.cat([
        x.data[lo:hi]
        for lo, hi in zip(lo_indexes, hi_indexes)
    ], dim=0)

def _require_enforce_sorted(x):
    if x.sorted_indices is not None:
        raise ValueError(
            'this function can only be used on packed sequences that were '
            'created with enforce_sorted=True')
