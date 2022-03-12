import itertools
import math
import unittest

import numpy
import torch

from torch_rnn_tools import UnidirectionalLSTM
from nsrnn.models.nondeterministic_stack import NondeterministicStackRNN
from nsrnn.semiring import real, log
from reference_util import recursively_stack

class TestNondeterministicStackRNN(unittest.TestCase):

    def test_forward_and_backward(self):
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = NondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.MSELoss()
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        predicted_tensor = model(input_tensor, block_size=10)
        self.assertEqual(
            predicted_tensor.size(),
            (batch_size, sequence_length, hidden_units),
            'output does not have expected dimensions'
        )
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        loss = criterion(predicted_tensor, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_against_reference(self):
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 6
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = NondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        reference_model = ReferenceNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p, reference_p in itertools.zip_longest(model.parameters(), reference_model.parameters()):
            reference_p.data.copy_(p.data)
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        predicted_tensor = model(input_tensor, block_size=10)
        reference_predicted_tensor = reference_model(input_tensor)
        numpy.testing.assert_allclose(
            predicted_tensor.detach(),
            reference_predicted_tensor.detach(),
            rtol=1e-4,
            err_msg='output of model does not agree with reference implementation')
        criterion = torch.nn.MSELoss()
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        loss = criterion(predicted_tensor, target_tensor)
        loss.backward()
        reference_loss = criterion(reference_predicted_tensor, target_tensor)
        reference_loss.backward()
        for p, reference_p in itertools.zip_longest(model.parameters(), reference_model.parameters()):
            numpy.testing.assert_allclose(
                p.grad,
                reference_p.grad,
                rtol=1e-3,
                err_msg='gradient does not agree with reference implementation')

class ReferenceNondeterministicStackRNN(NondeterministicStackRNN):

    def initial_stack(self, batch_size, stack_size, sequence_length):
        assert self.normalize_reading
        assert self.include_states_in_reading
        tensor = next(self.parameters())
        return ReferenceNondeterministicStack(
            batch_size,
            self.num_states,
            self.stack_alphabet_size,
            tensor.device
        )

class RealSemiring:

    @property
    def semiring(self):
        return real

    def normalize(self, x, dim):
        return x / x.sum(dim=dim, keepdim=True)

    def convert_operation_weights(self, x):
        return torch.exp(x)

class LogSemiring:

    @property
    def semiring(self):
        return log

    def normalize(self, x, dim):
        return torch.nn.functional.softmax(x, dim=dim)

    def convert_operation_weights(self, x):
        return x

class ReferenceNondeterministicStack(RealSemiring):

    def __init__(self, B, Q, S, device):
        super().__init__()
        # self.gamma : B x {i} x {j} x {q} x {x} x {r} x {y}
        self.device = device
        self.gamma = self.semiring.zeros((B, 1, 1, Q, S, Q, S), device=device)
        self.alpha = self.semiring.zeros((B, 1, Q, S), device=device)
        self.alpha[:, 0, 0, 0] = self.semiring.one

    def get(self, container, key):
        try:
            return container[key]
        except IndexError:
            return self.semiring.zeros((), device=self.device)

    def sum(self, tensors):
        if tensors:
            return self.semiring.sum(torch.stack(tensors), dim=0)
        else:
            return self.semiring.zeros((), device=self.device)

    def update(self, push, repl, pop):
        B, j, Q, S = self.alpha.size()
        push = self.convert_operation_weights(push)
        repl = self.convert_operation_weights(repl)
        pop = self.convert_operation_weights(pop)
        gamma_j = recursively_stack([
            [
                [
                    [
                        [
                            [
                                self.sum([
                                    push[b, q, x, r, y] if i == j-1 else self.semiring.zeros((), device=self.device),
                                    self.sum([
                                        self.semiring.multiply(
                                            self.get(self.gamma, (b, i, j-1, q, x, s, z)),
                                            repl[b, s, z, r, y]
                                        )
                                        for s in range(Q)
                                        for z in range(S)
                                    ]),
                                    self.sum([
                                        self.semiring.multiply(
                                            self.semiring.multiply(
                                                self.get(self.gamma, (b, i, k, q, x, t, y)),
                                                self.get(self.gamma, (b, k, j-1, t, y, s, z))
                                            ),
                                            pop[b, s, z, r]
                                        )
                                        for k in range(i+1, j-2+1)
                                        for t in range(Q)
                                        for s in range(Q)
                                        for z in range(S)
                                    ])
                                ])
                                for y in range(S)
                            ]
                            for r in range(Q)
                        ]
                        for x in range(S)
                    ]
                    for q in range(Q)
                ]
                for i in range(j+1)
            ]
            for b in range(B)
        ])
        self.gamma = torch.cat([
            torch.cat([
                self.gamma,
                self.semiring.zeros((B, 1, j, Q, S, Q, S), device=self.device)
            ], dim=1),
            gamma_j[:, :, None]
        ], dim=2)
        alpha_j = recursively_stack([
            [
                [
                    self.sum([
                        self.semiring.multiply(
                            self.alpha[b, i, q, x],
                            self.gamma[b, i, j, q, x, r, y]
                        )
                        for i in range(0, j-1+1)
                        for q in range(Q)
                        for x in range(S)
                    ])
                    for y in range(S)
                ]
                for r in range(Q)
            ]
            for b in range(B)
        ])
        self.alpha = torch.cat([
            self.alpha,
            alpha_j[:, None]
        ], dim=1)

    def reading(self):
        B, _, Q, S = self.alpha.size()
        return self.normalize(recursively_stack([
            [
                self.alpha[b, -1, r, y]
                for r in range(Q)
                for y in range(S)
            ]
            for b in range(B)
        ]), dim=1)

if __name__ == '__main__':
    unittest.main()
