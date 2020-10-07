import unittest

import torch

from nsrnn.pytorch_tools.layer import Layer
from nsrnn.pytorch_tools.unidirectional_rnn import (
    UnidirectionalRNN, UnidirectionalLSTM, UnidirectionalNonRecurrent)

class TestUnidirectionalRNN(unittest.TestCase):

    def test_rnn(self):
        self._test_rnn_class(UnidirectionalRNN, layers=1)

    def test_multi_layer_rnn(self):
        self._test_rnn_class(UnidirectionalRNN, layers=3)

    def test_lstm(self):
        self._test_rnn_class(UnidirectionalLSTM, layers=1)

    def test_multi_layer_lstm(self):
        self._test_rnn_class(UnidirectionalLSTM, layers=3)

    def test_nonrecurrent(self):
        def constructor(input_size, hidden_units):
            return UnidirectionalNonRecurrent(Layer(input_size, hidden_units))
        self._test_rnn_class(constructor)

    def _test_rnn_class(self, constructor, **kwargs):
        input_size = 5
        hidden_units = 7
        batch_size = 11
        sequence_length = 13
        generator = torch.manual_seed(0)
        model = constructor(input_size=input_size, hidden_units=hidden_units,
            **kwargs)
        for p in model.parameters():
            p.data.uniform_(generator=generator)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        loss = 0
        state = model.initial_state(batch_size)
        for i in range(sequence_length):
            predicted_tensor = state.output()
            self.assertEqual(
                predicted_tensor.size(),
                (batch_size, hidden_units),
                'output has the expected dimensions'
            )
            target_tensor = torch.empty(batch_size, hidden_units)
            target_tensor.uniform_(generator=generator)
            loss += criterion(predicted_tensor, target_tensor)
            input_tensor = torch.empty(batch_size, input_size)
            input_tensor.uniform_(generator=generator)
            state = state.next(input_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    unittest.main()
