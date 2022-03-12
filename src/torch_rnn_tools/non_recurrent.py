import torch

from .unidirectional_rnn import UnidirectionalRNN

class NonRecurrentRNN(UnidirectionalRNN):
    """A dummy RNN that has no recurrent connections."""

    def __init__(self, model: torch.nn.Module):
        """
        :param model: A module that will be used to transform the input to the
            output at each time step. For example, this could be a
            feed-forward network. This module should have ``input_size()`` and
            ``output_size()`` methods. The first dimension of the input and
            output should be the batch dimension.
        """
        super().__init__()
        self.model = model

    def input_size(self):
        return self.model.input_size()

    def output_size(self):
        return self.model.output_size()

    class State(UnidirectionalRNN.State):

        def __init__(self, rnn, output):
            self.rnn = rnn
            self._output = output

        def next(self, input_tensor):
            output = self.rnn.model(input_tensor)
            return self.rnn.State(self.rnn, output)

        def output(self):
            return self._output

        def transform_tensors(self, func):
            return self.rnn.State(self.rnn, func(self._output))

        def batch_size(self):
            return self._output.size(0)

    def initial_state(self, batch_size):
        # Use 0 as the initial output.
        return self.State(
            self,
            torch.zeros(
                batch_size,
                self.output_size(),
                device=self.device
            ))
