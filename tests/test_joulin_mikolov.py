import unittest

import numpy
import torch

from nsrnn.models.joulin_mikolov import JoulinMikolovRNN

class TestJoulinMikolovRNN(unittest.TestCase):

    def test_synchronized_joulin_mikolov(self):
        self._test_joulin_mikolov(synchronized=True)

    def test_unsynchronized_joulin_mikolov(self):
        self._test_joulin_mikolov(synchronized=False)

    def _test_joulin_mikolov(self, synchronized):
        stack_embedding_size = 3
        input_size = 5
        hidden_units = 7
        batch_size = 11
        sequence_length = 13
        generator = torch.manual_seed(0)
        model = JoulinMikolovRNN(
            input_size=input_size,
            hidden_units=hidden_units,
            stack_embedding_size=stack_embedding_size,
            synchronized=synchronized
        )
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

    def test_even_length_hint(self):
        self._test_length_hint(20)

    def test_odd_length_hint(self):
        self._test_length_hint(21)

    def _test_length_hint(self, sequence_length):
        stack_embedding_size = 3
        input_size = 5
        hidden_units = 7
        batch_size = 11
        device = torch.device('cpu')
        generator = torch.manual_seed(0)
        model = JoulinMikolovRNN(
            input_size=input_size,
            hidden_units=hidden_units,
            stack_embedding_size=stack_embedding_size,
            synchronized=True
        )
        for name, p in model.named_parameters():
            p.data.uniform_(generator=generator)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(generator=generator)
        predicted_tensor_with_length = model(input_tensor)
        self.assertEqual(
            predicted_tensor_with_length.size(),
            (batch_size, sequence_length + 1, hidden_units))
        state_no_length = model.initial_state(batch_size, device)
        outputs_no_length = list(state_no_length.generate_outputs(input_tensor.transpose(0, 1)))
        predicted_tensor_no_length = torch.stack(outputs_no_length, dim=1)
        self.assertEqual(
            predicted_tensor_no_length.size(),
            (batch_size, sequence_length + 1, hidden_units))
        numpy.testing.assert_allclose(
            predicted_tensor_with_length.detach(),
            predicted_tensor_no_length.detach())
        predicted_tensor_reference = model(
            input_tensor,
            stack_constructor=construct_reference_stack
        )
        numpy.testing.assert_allclose(
            predicted_tensor_with_length.detach(),
            predicted_tensor_reference.detach())

    def test_against_reference(self):
        stack_embedding_size = 3
        input_size = 5
        hidden_units = 7
        batch_size = 11
        sequence_length = 20
        generator = torch.manual_seed(0)
        model = JoulinMikolovRNN(
            input_size=input_size,
            hidden_units=hidden_units,
            stack_embedding_size=stack_embedding_size,
            synchronized=True
        )
        for name, p in model.named_parameters():
            p.data.uniform_(generator=generator)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(generator=generator)
        predicted_tensor = model(input_tensor)
        predicted_tensor_reference = model(
            input_tensor,
            stack_constructor=construct_reference_stack
        )
        numpy.testing.assert_allclose(
            predicted_tensor.detach(),
            predicted_tensor_reference.detach())

    def test_return_signals(self):
        stack_embedding_size = 3
        input_size = 5
        hidden_units = 7
        batch_size = 11
        sequence_length = 8
        generator = torch.manual_seed(0)
        model = JoulinMikolovRNN(
            input_size=input_size,
            hidden_units=hidden_units,
            stack_embedding_size=stack_embedding_size,
            synchronized=True
        )
        for name, p in model.named_parameters():
            p.data.uniform_(generator=generator)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(generator=generator)
        predicted_tensor, signals = model(input_tensor, return_signals=True)
        self.assertIsInstance(predicted_tensor, torch.Tensor)
        self.assertEqual(
            predicted_tensor.size(),
            (batch_size, sequence_length + 1, hidden_units),
            'output has the expected dimensions'
        )
        self.assertIsInstance(signals, list)
        self.assertEqual(len(signals), sequence_length + 1)
        self.assertIsNone(signals[0])
        for signal in signals[1:]:
            self.assertIsInstance(signal, torch.Tensor)
            self.assertEqual(signal.size(), (batch_size, 1, 3))

class ReferenceJoulinMikolovStack:

    def __init__(self, elements, bottom):
        self.elements = elements
        self.bottom = bottom

    def reading(self):
        if self.elements:
            return self.elements[0]
        else:
            return self.bottom

    def next(self, push_prob, pop_prob, noop_prob, push_value):
        return ReferenceJoulinMikolovStack(
            list(self.next_elements(push_prob, pop_prob, noop_prob, push_value)),
            self.bottom
        )

    def next_elements(self, push_prob, pop_prob, noop_prob, push_value):
        push_prob = push_prob.squeeze(-1)
        pop_prob = pop_prob.squeeze(-1)
        noop_prob = noop_prob.squeeze(-1)
        new_stack_height = len(self.elements) + 1
        for i in range(new_stack_height):
            if i > 0:
                above = self.elements[i-1]
            else:
                above = push_value
            if i < new_stack_height - 1:
                same = self.elements[i]
            else:
                same = self.bottom
            if i < new_stack_height - 2:
                below = self.elements[i+1]
            else:
                below = self.bottom
            yield push_prob * above + noop_prob * same + pop_prob * below

def construct_reference_stack(batch_size, stack_size, sequence_length, device):
    return ReferenceJoulinMikolovStack(
        [],
        torch.zeros(batch_size, stack_size, device=device)
    )

if __name__ == '__main__':
    unittest.main()
