import torch

from ..util import to_list
from ._common import apply_to_first_element

class UnidirectionalRNNBase(torch.nn.Module):

    def forward(self, input_sequence, *args, only_last=False, **kwargs):
        if isinstance(input_sequence, tuple):
            input_sequence, lengths = input_sequence
        elif only_last:
            raise ValueError('only_last=True requires providing sequence lengths')
        # input_sequence: batch_size x sequence_length x ...
        inputs = input_sequence.transpose(0, 1)
        outputs = self.generate_outputs(inputs, *args, **kwargs)
        result = self.stack_outputs(outputs)
        if only_last:
            result = apply_to_first_element(
                lambda x: get_last_outputs(x, lengths),
                result
            )
        return result

    def generate_outputs(self, input_tensors, *args, **kwargs):
        # input_tensors: sequence_length x batch_size x ...
        batch_size = input_tensors.size(1)
        device = input_tensors.device
        state = self.initial_state(batch_size, device, *args, **kwargs)
        return state.generate_outputs(input_tensors)

    def stack_outputs(self, outputs):
        it = iter(outputs)
        first = next(it)
        if isinstance(first, tuple):
            output, *extra = first
            output_list = [output]
            extra_lists = tuple([e] for e in extra)
            for output, *extra in it:
                output_list.append(output)
                for extra_list, extra_item in zip(extra_lists, extra):
                    extra_list.append(extra_item)
            return (torch.stack(output_list, dim=1), *extra_lists)
        else:
            output_list = [first]
            output_list.extend(it)
            return torch.stack(output_list, dim=1)

    def initial_state(self, batch_size, device=None):
        raise NotImplementedError

    def input_size(self):
        raise NotImplementedError

    def output_size(self):
        raise NotImplementedError

    def wrapped_rnn(self):
        return self

    class State:

        def next(self, input_tensor):
            raise NotImplementedError

        def output(self):
            raise NotImplementedError

        def fastforward(self, input_sequence):
            # input_sequence: batch_size x sequence_length x ...
            input_tensors = input_sequence.transpose(0, 1)
            state = self
            for input_tensor in input_tensors:
                state = state.next(input_tensor)
            return state

        def outputs(self, input_sequence):
            # input_sequence: batch_size x sequence_length x ...
            inputs = input_sequence.transpose(0, 1)
            outputs = list(self.generate_outputs(inputs))
            return torch.stack(outputs, dim=1)

        def generate_outputs(self, input_tensors):
            state = self
            yield state.output()
            for input_tensor in input_tensors:
                state = state.next(input_tensor)
                yield state.output()

def construct_lengths_tensor(lengths, device):
    lengths = to_list(lengths)
    return torch.tensor(lengths, device=device).unsqueeze(1).unsqueeze(2)

def get_last_outputs(tensor, lengths):
    h = tensor.size(2)
    return torch.gather(tensor, 1, lengths.expand(-1, -1, h)).squeeze(1)

class UnidirectionalRNNImplBase(UnidirectionalRNNBase):

    def __init__(self, input_size, hidden_units, layers=1, dropout=None,
            **kwargs):
        if dropout is None:
            dropout = 0
        super().__init__()
        RNNClass = self.rnn_class()
        self.rnn = RNNClass(input_size, hidden_units,
            num_layers=layers, batch_first=False, bidirectional=False,
            dropout=dropout, **kwargs)
        self._input_size = input_size
        self._hidden_units = hidden_units
        self._layers = layers

    def rnn_class(self):
        raise NotImplementedError

    def initial_values(self, batch_size, device):
        raise NotImplementedError

    def hidden_state_to_output(self, hidden_state):
        raise NotImplementedError

    def initial_state(self, batch_size, device=None):
        hidden_state, output = self.initial_values(batch_size, device)
        return self.State(self, hidden_state, output)

    def input_size(self):
        return self._input_size

    def output_size(self):
        return self._hidden_units

    class State(UnidirectionalRNNBase.State):

        def __init__(self, rnn, hidden_state, output):
            self.rnn = rnn
            self.hidden_state = hidden_state
            self._output = output

        def next(self, input_tensor):
            output, hidden_state = self.rnn.rnn(
                input_tensor.unsqueeze(dim=0), self.hidden_state)
            return self.rnn.State(
                self.rnn, hidden_state, output.squeeze(dim=0))

        def output(self):
            return self._output

class UnidirectionalRNN(UnidirectionalRNNImplBase):

    def rnn_class(self):
        return torch.nn.RNN

    def initial_values(self, batch_size, device):
        zero = torch.zeros(
            self._layers, batch_size, self._hidden_units, device=device)
        return zero, zero[0]

class UnidirectionalLSTM(UnidirectionalRNNImplBase):

    def rnn_class(self):
        return torch.nn.LSTM

    def initial_values(self, batch_size, device):
        zero = torch.zeros(
            self._layers, batch_size, self._hidden_units, device=device)
        return (zero, zero), zero[0]

class UnidirectionalNonRecurrent(UnidirectionalRNNBase):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def initial_state(self, batch_size, device=None):
        return self.State(self,
            torch.zeros(batch_size, self.output_size(), device=device))

    def input_size(self):
        return self.model.input_size()

    def output_size(self):
        return self.model.output_size()

    class State(UnidirectionalRNNBase.State):

        def __init__(self, rnn, output):
            self.rnn = rnn
            self._output = output

        def next(self, input_tensor):
            output = self.rnn.model(input_tensor)
            return self.rnn.State(self.rnn, output)

        def output(self):
            return self._output
