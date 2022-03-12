import attr
import torch

from torch_rnn_tools import UnidirectionalRNN

class StackRNNBase(UnidirectionalRNN):

    def __init__(self, input_size, stack_reading_size, controller):
        super().__init__()
        self._input_size = input_size
        self.stack_reading_size = stack_reading_size
        self.controller = controller(input_size + stack_reading_size)

    def input_size(self):
        return self._input_size

    def output_size(self):
        return self.controller.output_size()

    class State(UnidirectionalRNN.State):

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

        def detach(self):
            return self.rnn.State(
                self.rnn,
                self.state.detach(),
                self.stack.detach() if self.stack is not None else None,
                self.previous_stack.detach() if self.previous_stack is not None else None,
                self.return_signals
            )

        def batch_size(self):
            return self.state.batch_size()

        def slice_batch(self, s):
            return self.rnn.State(
                self.rnn,
                self.state.slice_batch(s),
                self.stack.slice_batch(s) if self.stack is not None else None,
                self.previous_stack.slice_batch(s) if self.previous_stack is not None else None,
                self.return_signals
            )

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

    def initial_state(self, batch_size, *args, return_signals=False, **kwargs):
        return self.State(
            self,
            self.controller.initial_state(batch_size),
            self.initial_stack(
                batch_size,
                self.stack_reading_size,
                *args,
                **kwargs),
            None,
            return_signals
        )

    def initial_stack(self, batch_size, stack_size, *args, **kwargs):
        raise NotImplementedError
