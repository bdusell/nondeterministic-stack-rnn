import unittest

import torch
import numpy

from nsrnn.pytorch_tools.rnn import RNN, LSTM

class TestRNN(unittest.TestCase):

    def test_rnn(self):
        self._test_rnn_class(RNN)

    def test_lstm(self):
        self._test_rnn_class(LSTM)

    def _test_rnn_class(self, constructor):
        input_size = 5
        hidden_units = 7
        batch_size = 11
        sequence_length = 13
        generator = torch.manual_seed(0)
        model = constructor(input_size=input_size, hidden_units=hidden_units)
        for p in model.parameters():
            p.data.uniform_(generator=generator)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(generator=generator)
        predicted_tensor = model(input_tensor)
        self.assertEqual(
            predicted_tensor.size(),
            (batch_size, sequence_length, hidden_units),
            'output has the expected dimensions'
        )
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(generator=generator)
        loss = criterion(predicted_tensor, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_reusing_state(self):
        constructor = LSTM
        input_size = 2
        hidden_units = 3
        batch_size = 7
        sequence_length = 8
        layers = 4
        generator = torch.manual_seed(0)
        model = constructor(input_size=input_size, hidden_units=hidden_units,
            layers=layers)
        for p in model.parameters():
            p.data.uniform_(generator=generator)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(generator=generator)
        predicted_tensor, state = model(input_tensor,
            initial_state=None, return_state=True)
        self.assertEqual(
            predicted_tensor.size(),
            (batch_size, sequence_length, hidden_units))
        state = state.detach()
        self.assertEqual(
            state.value[0].size(),
            (layers, batch_size, hidden_units))
        state_1 = state.value[0][-1]
        self.assertEqual(
            state_1.size(),
            (batch_size, hidden_units))
        output_tensor, state = model(input_tensor,
            initial_state=state, return_state=True)
        state = state.detach()
        state_2 = output_tensor[:, 0]
        self.assertEqual(
            state_2.size(),
            (batch_size, hidden_units))
        numpy.testing.assert_allclose(state_1.detach(), state_2.detach())

    def test_iterating_states(self):
        constructor = LSTM
        input_size = 2
        hidden_units = 3
        batch_size = 7
        sequence_length = 11
        layers = 4
        generator = torch.manual_seed(0)
        device = torch.device('cpu')
        model = constructor(input_size=input_size, hidden_units=hidden_units,
            layers=layers)
        model.to(device)
        for p in model.parameters():
            p.data.uniform_(generator=generator)
        input_tensor = torch.empty(
            batch_size,
            sequence_length - 1,
            input_size,
            device=device)
        input_tensor.uniform_(generator=generator)
        whole_output_tensor = model(input_tensor)
        self.assertEqual(
            whole_output_tensor.size(),
            (batch_size, sequence_length, hidden_units))
        state = model.initial_state(batch_size)
        yt_expected = whole_output_tensor[:, 0]
        yt_test = state.output()
        self.assertEqual(yt_test.size(), (batch_size, hidden_units))
        numpy.testing.assert_allclose(yt_test.detach(), yt_expected.detach())
        for i in range(sequence_length - 1):
            xt = input_tensor[:, i]
            state = state.next(xt)
            yt_expected = whole_output_tensor[:, i+1]
            yt_test = state.output()
            numpy.testing.assert_allclose(yt_test.detach(), yt_expected.detach(), rtol=1e-6)

if __name__ == '__main__':
    unittest.main()
