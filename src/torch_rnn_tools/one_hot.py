import torch

from .unidirectional_rnn import UnidirectionalRNN
from .wrapper import Wrapper

class OneHotWrapper(Wrapper):
    """Converts indexes to one-hot vectors in the input to an RNN."""

    def transform_input(self, x):
        # x : batch_size (x sequence_length) in [0, self.rnn.input_size())
        # result : batch_size (x sequence_length) x self.rnn.input_size()
        result = x.new_zeros((*x.size(), self.rnn.input_size()), dtype=torch.float)
        result.scatter_(-1, x[..., None], 1)
        return result

    def input_size(self):
        raise TypeError('OneHotWrapper does not have an input_size')
