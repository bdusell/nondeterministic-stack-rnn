import itertools
import math
import unittest

import numpy
import torch

from torch_rnn_tools import UnidirectionalLSTM
from nsrnn.models.nondeterministic_stack import NondeterministicStackRNN
from nsrnn.models.limited_nondeterministic_stack import (
    LimitedNondeterministicStackRNN,
    Operation,
    PushOperation,
    ReplaceOperation,
    PopOperation
)
from nsrnn.semiring import log
from reference_util import recursively_stack
from test_nondeterministic_stack_rnn import (
    ReferenceNondeterministicStackRNN,
    ReferenceNondeterministicStack,
    LogSemiring
)

class TestLimitedNondeterministicStackRNN(unittest.TestCase):

    def test_forward_and_backward(self):
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
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
        for name, p in model.named_parameters():
            self.assertFalse(
                torch.any(torch.isnan(p.grad)).item(),
                f'gradient for parameter {name} has nan')
        optimizer.step()

    def test_forward_and_backward_against_reference(self):
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        reference_model = ReferenceLimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p, reference_p in itertools.zip_longest(model.parameters(), reference_model.parameters()):
            reference_p.data.copy_(p.data)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.MSELoss()
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        predicted_tensor = model(input_tensor, block_size=10)
        reference_predicted_tensor = reference_model(input_tensor)
        numpy.testing.assert_allclose(
            predicted_tensor.detach(),
            reference_predicted_tensor.detach(),
            rtol=1e-4,
            err_msg='output of forward() does not match reference implementation')
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
                err_msg='gradient does not match reference implementation')

    def test_truncated_bptt(self):
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.MSELoss()
        state = model.initial_state(batch_size)

        # First iteration
        optimizer.zero_grad()
        self.assertEqual(state.batch_size(), batch_size)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        predicted_tensor, state = model(
            input_tensor,
            initial_state=state,
            return_state=True,
            include_first=False,
            block_size=10
        )
        state = state.detach()
        self.assertEqual(
            predicted_tensor.size(),
            (batch_size, sequence_length, hidden_units),
            'output does not have expected dimensions'
        )
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        loss = criterion(predicted_tensor, target_tensor)
        loss.backward()
        optimizer.step()

        # Second iteration with smaller batch size.
        self.assertEqual(state.batch_size(), batch_size)
        batch_size -= 1
        state = state.slice_batch(slice(batch_size))
        self.assertEqual(state.batch_size(), batch_size)
        optimizer.zero_grad()
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        predicted_tensor, state = model(
            input_tensor,
            initial_state=state,
            return_state=True,
            include_first=False,
            block_size=10
        )
        state = state.detach()
        self.assertEqual(
            predicted_tensor.size(),
            (batch_size, sequence_length, hidden_units),
            'output does not have expected dimensions'
        )
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        loss = criterion(predicted_tensor, target_tensor)
        loss.backward()
        optimizer.step()

    def test_reference_limited_matches_reference_unlimited(self):
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 6
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        limited_model = ReferenceLimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=sequence_length,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p in limited_model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        unlimited_model = ReferenceNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for unlimited_p, limited_p in itertools.zip_longest(unlimited_model.parameters(), limited_model.parameters()):
            unlimited_p.data.copy_(limited_p.data)
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        limited_predicted_tensor = limited_model(input_tensor)
        unlimited_predicted_tensor = unlimited_model(input_tensor)
        numpy.testing.assert_allclose(
            limited_predicted_tensor.detach(),
            unlimited_predicted_tensor.detach(),
            err_msg='output of limited reference implementation does not '
                    'match output of unlimited reference implementation')
        criterion = torch.nn.MSELoss()
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        limited_loss = criterion(limited_predicted_tensor, target_tensor)
        limited_loss.backward()
        unlimited_loss = criterion(unlimited_predicted_tensor, target_tensor)
        unlimited_loss.backward()
        for limited_p, unlimited_p in itertools.zip_longest(limited_model.parameters(), unlimited_model.parameters()):
            numpy.testing.assert_allclose(
                limited_p.grad,
                unlimited_p.grad,
                err_msg='gradient of limited reference implementation does '
                        'not match gradient of unlimited reference '
                        'implementation')

    def test_large_window_size(self):
        # Test that the model works when window_size (D) is larger than the
        # sequence length.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 6
        window_size = 20
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        limited_model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p in limited_model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        unlimited_model = NondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for unlimited_p, limited_p in itertools.zip_longest(unlimited_model.parameters(), limited_model.parameters()):
            unlimited_p.data.copy_(limited_p.data)
        input_tensor = torch.empty(batch_size, sequence_length - 1, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        limited_predicted_tensor = limited_model(input_tensor, block_size=10)
        unlimited_predicted_tensor = unlimited_model(input_tensor, block_size=10)
        numpy.testing.assert_allclose(
            limited_predicted_tensor.detach(),
            unlimited_predicted_tensor.detach(),
            rtol=1e-4,
            err_msg='output of limited reference implementation does not '
                    'match output of unlimited reference implementation')
        criterion = torch.nn.MSELoss()
        target_tensor = torch.empty(batch_size, sequence_length, hidden_units)
        target_tensor.uniform_(-1, 1, generator=generator)
        limited_loss = criterion(limited_predicted_tensor, target_tensor)
        limited_loss.backward()
        unlimited_loss = criterion(unlimited_predicted_tensor, target_tensor)
        unlimited_loss.backward()
        for limited_p, unlimited_p in itertools.zip_longest(limited_model.parameters(), unlimited_model.parameters()):
            numpy.testing.assert_allclose(
                limited_p.grad,
                unlimited_p.grad,
                rtol=1e-3,
                err_msg='gradient of limited reference implementation does '
                        'not match gradient of unlimited reference '
                        'implementation')

    def test_chunks_match_whole(self):
        # Check that processing a sequence in chunks produces the same
        # mathematical result as processing it all at once.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        num_chunks = 3
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)

        input_tensor = torch.empty(batch_size, sequence_length * num_chunks, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)

        expected_output, _ = model(
            input_tensor,
            initial_state=model.initial_state(batch_size),
            return_state=True,
            include_first=False,
            block_size=10
        )

        predicted_outputs = []
        state = model.initial_state(batch_size)
        for chunk_no in range(num_chunks):
            offset = chunk_no * sequence_length
            predicted_output, state = model(
                input_tensor[:, offset:offset+sequence_length],
                initial_state=state,
                return_state=True,
                include_first=False,
                block_size=10
            )
            predicted_outputs.append(predicted_output)
            state = state.detach()
        predicted_output = torch.cat(predicted_outputs, dim=1)

        numpy.testing.assert_allclose(
            predicted_output.detach(),
            expected_output.detach(),
            err_msg='output of chunks does not match output of whole sequence')

    def test_large_window_size_chunks(self):
        # Check that processing chunks still works correctly when the
        # window_size (D) is longer than the chunk length.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        chunk_lengths = [3, 4, 10, 5]
        window_size = 8
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)

        input_tensor = torch.empty(batch_size, sum(chunk_lengths), input_size)
        input_tensor.uniform_(-1, 1, generator=generator)

        expected_output, _ = model(
            input_tensor,
            initial_state=model.initial_state(batch_size),
            return_state=True,
            include_first=False,
            block_size=10
        )

        predicted_outputs = []
        state = model.initial_state(batch_size)
        for chunk in input_tensor.split(chunk_lengths, dim=1):
            predicted_output, state = model(
                chunk,
                initial_state=state,
                return_state=True,
                include_first=False,
                block_size=10
            )
            predicted_outputs.append(predicted_output)
            state = state.detach()
        predicted_output = torch.cat(predicted_outputs, dim=1)

        numpy.testing.assert_allclose(
            predicted_output.detach(),
            expected_output.detach(),
            err_msg='output of chunks does not match output of whole sequence')

    def test_return_signals(self):
        # Test that the model can return the operation weights at every time
        # step.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        B = batch_size
        T = sequence_length
        H = hidden_units
        Q = num_states
        S = stack_alphabet_size
        model.eval()
        with torch.no_grad():

            output, signals = model(
                input_tensor,
                include_first=False,
                return_signals=True,
                block_size=10
            )
            self.assertEqual(output.size(), (B, T, H))
            self.assertEqual(len(signals), T)
            for signal in signals:
                push, repl, pop = signal
                self.assertEqual(push.size(), (B, Q, S, Q, S))
                self.assertEqual(repl.size(), (B, Q, S, Q, S))
                self.assertEqual(pop.size(), (B, Q, S, Q))

            (output, signals), state = model(
                input_tensor,
                include_first=False,
                return_signals=True,
                return_state=True,
                block_size=10
            )
            self.assertEqual(output.size(), (B, T, H))
            self.assertEqual(len(signals), T)
            for signal in signals:
                push, repl, pop = signal
                self.assertEqual(push.size(), (B, Q, S, Q, S))
                self.assertEqual(repl.size(), (B, Q, S, Q, S))
                self.assertEqual(pop.size(), (B, Q, S, Q))
            self.assertIsInstance(state, LimitedNondeterministicStackRNN.InitialState)

    def test_viterbi(self):
        # Test that the Viterbi algorithm runs without error.
        batch_size = 5
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        sequence_length = 13
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)
        input_tensor = torch.empty(batch_size, sequence_length, input_size)
        input_tensor.uniform_(-1, 1, generator=generator)
        model.eval()
        paths, scores = model.viterbi_decoder(input_tensor, block_size=10).decode_timestep(sequence_length)
        for path in paths:
            for operation in path:
                self.assertIsInstance(operation, Operation)
        self.assertEqual(scores.size(), (batch_size,))

    def test_hardcoded_best_runs(self):
        # Test that the output of the Viterbi algorithm is correct.
        input_size = 7
        num_states = 2
        stack_alphabet_size = 3
        hidden_units = 11
        window_size = 5
        generator = torch.manual_seed(123)
        def controller(input_size):
            return UnidirectionalLSTM(input_size, hidden_units)
        model = LimitedNondeterministicStackRNN(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            window_size=window_size,
            controller=controller,
            normalize_operations=False,
            normalize_reading=True,
            include_states_in_reading=True
        )
        for p in model.parameters():
            p.data.uniform_(-0.1, 0.1, generator=generator)

        expected_run = [
            PushOperation(state_to=1, symbol=1),
            ReplaceOperation(state_to=0, symbol=2),
            PushOperation(state_to=1, symbol=0),
            PopOperation(state_to=0),
            ReplaceOperation(state_to=1, symbol=1),
            PushOperation(state_to=0, symbol=2)
        ]

        batch_size = 1
        B = batch_size
        Q = num_states
        S = stack_alphabet_size
        operation_weights = []
        for operation in expected_run:
            push = log.zeros((B, Q, S, Q, S))
            repl = log.zeros((B, Q, S, Q, S))
            pop = log.zeros((B, Q, S, Q))
            if type(operation) is PushOperation:
                push[:, :, :, operation.state_to, operation.symbol] = log.one
            elif type(operation) is ReplaceOperation:
                repl[:, :, :, operation.state_to, operation.symbol] = log.one
            elif type(operation) is PopOperation:
                pop[:, :, :, operation.state_to] = log.one
            operation_weights.append((push, repl, pop))

        model.eval()
        with torch.no_grad():
            (path,), scores = \
                model.viterbi_decoder_from_operation_weights(operation_weights, block_size=10) \
                    .decode_timestep(len(operation_weights))
        self.assertEqual(list(path), expected_run)
        self.assertEqual(scores.size(), (1,))
        numpy.testing.assert_allclose(scores, torch.tensor([0.0]))

class ReferenceLimitedNondeterministicStackRNN(NondeterministicStackRNN):

    def __init__(self, *args, window_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size

    def initial_stack(self, batch_size, stack_size, sequence_length):
        assert self.normalize_reading
        tensor = next(self.parameters())
        return ReferenceLimitedNondeterministicStack(
            batch_size,
            self.num_states,
            self.stack_alphabet_size,
            self.window_size,
            tensor.device
        )

class ReferenceLimitedNondeterministicStack(ReferenceNondeterministicStack):

    def __init__(self, B, Q, S, D, device):
        super().__init__(B, Q, S, device)
        self.D = D

    def update(self, push, repl, pop):
        B, j, Q, S = self.alpha.size()
        D = self.D
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
                                if i >= j-D
                                else self.semiring.zeros((), device=self.device)
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

if __name__ == '__main__':
    unittest.main()
