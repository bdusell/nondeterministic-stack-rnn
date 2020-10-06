import torch

from ..pytorch_tools.unidirectional_rnn import UnidirectionalRNNBase

class StackRNNBase(UnidirectionalRNNBase):

    def __init__(self, input_size, hidden_units, stack_size, controller):
        super().__init__()
        self._input_size = input_size
        self._hidden_units = hidden_units
        self.stack_size = stack_size
        self.controller = controller(input_size + stack_size, hidden_units)

    def input_size(self):
        return self._input_size

    def output_size(self):
        return self._hidden_units

    def initial_state(self, batch_size, device, *args, return_signals=False,
            **kwargs):
        return self.State(
            self,
            self.controller.initial_state(batch_size, device),
            self.initial_stack(batch_size, self.stack_size, device, *args, **kwargs),
            None,
            return_signals
        )

    def initial_stack(self, batch_size, stack_size, device, *args, **kwargs):
        raise NotImplementedError

    class State(UnidirectionalRNNBase.State):

        def __init__(self, rnn, state, stack, previous_stack, return_signals):
            super().__init__()
            self.rnn = rnn
            self.state = state
            self.stack = stack
            self.previous_stack = previous_stack
            self.return_signals = return_signals
            self._output = None
            self.signals = None

        def next(self, input_tensor):
            stack = self.get_stack()
            reading = stack.reading()
            controller_input = torch.cat((input_tensor, reading), dim=1)
            state = self.state.next(controller_input)
            return self.rnn.State(self.rnn, state, None, stack,
                self.return_signals)

        def output(self):
            output = self.get_output()
            if self.return_signals:
                self.get_stack()
                output = output, self.signals
            return output

        def get_output(self):
            if self._output is None:
                self._output = self.state.output()
            return self._output

        def get_stack(self):
            # This saves an unnecessary stack computation at the end of the
            # sequence, which can be rather expensive.
            if self.stack is None:
                hidden_state = self.get_output()
                self.stack = self.compute_stack(hidden_state, self.previous_stack)
            return self.stack

        def compute_stack(self, hidden_state, stack):
            raise NotImplementedError
