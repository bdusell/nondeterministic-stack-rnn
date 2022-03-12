import torch

from torch_extras.layer import Layer

from .unidirectional_rnn import UnidirectionalRNN
from .wrapper import Wrapper

class OutputWrapper(Wrapper):
    """Transforms the outputs of an RNN using a module."""

    def __init__(self, rnn: UnidirectionalRNN, layer: torch.nn.Module):
        """
        :param rnn: The wrapped RNN.
        :param layer: A module that will be applied to outputs of the RNN.
            This module should support an ``output_size()`` method.
        """
        super().__init__(rnn)
        self.layer = layer

    def transform_output(self, y):
        return self.layer(y)

    def output_size(self):
        return self.layer.output_size()

class OutputLayerWrapper(OutputWrapper):
    """Transforms the outputs of an RNN using a fully-connected layer."""

    def __init__(self, rnn: UnidirectionalRNN, *args, **kwargs):
        """Accepts the arguments of :py:meth:`torch_extras.layer.Layer.__init__`
        except for ``input_size``."""
        super().__init__(rnn, Layer(
            rnn.output_size(),
            *args,
            **kwargs
        ))
