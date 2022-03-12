import torch
from torch_semiring_einsum import compile_equation

from ..pytorch_tools.set_slice import set_slice
from ..semiring import log
from .common import StackRNNBase

class NondeterministicStackRNN(StackRNNBase):

    def __init__(self, input_size, num_states, stack_alphabet_size, controller,
            normalize_operations=True, normalize_reading=True,
            include_states_in_reading=False):
        Q = num_states
        S = stack_alphabet_size
        super().__init__(
            input_size=input_size,
            stack_reading_size=Q * S if include_states_in_reading else S,
            controller=controller
        )
        self.num_states = num_states
        self.stack_alphabet_size = stack_alphabet_size
        self.normalize_operations = normalize_operations
        self.normalize_reading = normalize_reading
        self.include_states_in_reading = include_states_in_reading
        self.num_op_rows = Q * S
        self.num_op_cols = Q * S + Q * S + Q
        self.operation_layer = torch.nn.Linear(
            self.controller.output_size(),
            self.num_op_rows * self.num_op_cols
        )

    def operation_log_scores(self, hidden_state):
        B = hidden_state.size(0)
        Q = self.num_states
        S = self.stack_alphabet_size
        # flat_logits : B x ((Q * S) * (Q * S + Q * S + Q))
        flat_logits = self.operation_layer(hidden_state)
        # logits : B x (Q * S) x (Q * S + Q * S + Q)
        logits = flat_logits.view(B, self.num_op_rows, self.num_op_cols)
        if self.normalize_operations:
            # Normalize the weights so that they sum to 1.
            logits = torch.nn.functional.log_softmax(logits, dim=2)
        push_chunk, repl_chunk, pop_chunk = logits.split([Q * S, Q * S, Q], dim=2)
        push = push_chunk.view(B, Q, S, Q, S)
        repl = repl_chunk.view(B, Q, S, Q, S)
        pop = pop_chunk.view(B, Q, S, Q)
        return push, repl, pop

    def forward(self, input_sequence, *args, return_signals=False, **kwargs):
        sequence_length = input_sequence.size(1)
        return super().forward(
            input_sequence,
            *args,
            # The last time step can be skipped unless returning the operation
            # weights.
            sequence_length=sequence_length - int(not return_signals),
            return_signals=return_signals,
            **kwargs)

    def initial_stack(self, batch_size, stack_size, sequence_length, block_size, semiring=log):
        tensor = next(self.parameters())
        return NondeterministicStack(
            batch_size=batch_size,
            num_states=self.num_states,
            stack_alphabet_size=self.stack_alphabet_size,
            sequence_length=sequence_length,
            normalize_reading=self.normalize_reading,
            include_states_in_reading=self.include_states_in_reading,
            block_size=block_size,
            dtype=tensor.dtype,
            device=tensor.device,
            semiring=semiring
        )

    class State(StackRNNBase.State):

        def compute_stack(self, hidden_state, stack):
            push, repl, pop = self.signals = self.rnn.operation_log_scores(hidden_state)
            stack.update(push, repl, pop)
            return stack

class NondeterministicStack:

    def __init__(self, batch_size, num_states, stack_alphabet_size,
            sequence_length, normalize_reading, include_states_in_reading,
            block_size, dtype, device, semiring):
        super().__init__()
        B = batch_size
        Q = num_states
        S = stack_alphabet_size
        n = sequence_length
        self.semiring = semiring
        self.alpha = semiring.zeros((B, n+1, Q, S), dtype=dtype, device=device)
        self.gamma = semiring.zeros((B, n, n, Q, S, Q, S), dtype=dtype, device=device)
        if not normalize_reading:
            # If the stack reading is not going to be normalized, do not use
            # -inf for the 0 weights in the initial time step, but use a
            # really negative number. This avoids nans.
            semiring.get_tensor(self.alpha)[:, 0, :, :] = -1e10
        semiring.get_tensor(self.alpha)[:, 0, 0, 0] = semiring.one
        self.block_size = block_size
        self.normalize_reading = normalize_reading
        self.include_states_in_reading = include_states_in_reading
        self.j = 0

    def update(self, push, repl, pop):
        # push : B x Q x S x Q x S
        # repl : B x Q x S x Q x S
        # pop : B x Q x S x Q
        semiring = self.semiring
        block_size = self.block_size
        self.j = j = self.j + 1
        gamma_j = next_gamma_column(
            self.gamma,
            j,
            push,
            repl,
            pop,
            semiring,
            block_size
        )
        self.gamma = semiring.combine(
            [self.gamma, gamma_j],
            lambda args: set_slice(
                args[0],
                (slice(None), slice(None, j), (j)-1),
                args[1]))
        alpha_j = next_alpha_column(
            self.alpha,
            gamma_j,
            j,
            semiring,
            block_size
        )
        self.alpha = semiring.combine(
            [self.alpha, alpha_j],
            lambda args: set_slice(
                args[0],
                (slice(None), j),
                args[1]))

    def reading(self):
        # Return log P_j(r, y).
        # alpha[0...j] has already been computed.
        semiring = self.semiring
        # alpha_j : B x Q x S
        alpha_j = self.alpha[:, self.j]
        if self.include_states_in_reading:
            B = alpha_j.size(0)
            # result : B x (Q * S)
            result = semiring.on_tensor(alpha_j, lambda x: x.view(B, -1))
        else:
            # result : B x S
            result = semiring.sum(alpha_j, dim=1)
        if self.normalize_reading:
            # Using softmax, normalize the log-weights so they sum to 1.
            assert semiring is log
            result = torch.nn.functional.softmax(result, dim=1)
        return result

def next_gamma_column(gamma, j, push, repl, pop, semiring, block_size):
    # gamma : B x n-1 x n-1 x Q x S x Q x S
    # return : B x j x Q x S x Q x S
    return combine_terms(
        push_term(push, semiring),
        repl_term(gamma, j, repl, semiring, block_size),
        pop_term(gamma, j, pop, semiring, block_size),
        semiring
    )

def combine_terms(push_term, repl_term, pop_term, semiring):
    # push_term : B x 1 x Q x S x Q x S
    # repl_term : B x max(0, j-1) x Q x S x Q x S
    # pop_term : B x max(0, j-2) x Q x S x Q x S
    # return : B x j x Q x S x Q x S
    return semiring.combine([
        semiring.add(
            semiring.on_tensor(repl_term, lambda x: x[:, :-1]),
            pop_term
        ),
        semiring.on_tensor(repl_term, lambda x: x[:, -1:]),
        push_term
    ], lambda args: torch.cat(args, dim=1))

def push_term(push, semiring):
    # push : B x Q x S x Q x S
    # return : B x 1 x Q x S x Q x S
    return semiring.on_tensor(push, lambda x: x[:, None, ...])

REPL_EQUATION = compile_equation('biqxsz,bszry->biqxry')

def repl_term(gamma, j, repl, semiring, block_size):
    # gamma : B x n-1 x n-1 x Q x S x Q x S
    # repl : B x Q x S x Q x S
    # return : B x (j-1) x Q x S x Q x S
    if j == 1:
        B, _, _, Q, S, _, _ = semiring.get_tensor(gamma).size()
        return semiring.primitive(
            semiring.get_tensor(gamma).new_empty(B, 0, Q, S, Q, S))
    else:
        # gamma[:, :j-1, j-2] : B x (j-1) x Q x S x Q x S
        return semiring.einsum(
            REPL_EQUATION,
            semiring.on_tensor(gamma, lambda x: x[:, :j-1, j-2]),
            repl,
            block_size=block_size
        )

POP_EQUATION = compile_equation('bikqxty,bktysz,bszr->biqxry')

def pop_term(gamma, j, pop, semiring, block_size):
    # gamma : B x n-1 x n-1 x Q x S x Q x S
    # pop : B x Q x S x Q
    # return : B x (j-2) x Q x S x Q x S
    if j <= 2:
        B, _, _, Q, S, _, _ = semiring.get_tensor(gamma).size()
        return semiring.primitive(
            semiring.get_tensor(gamma).new_empty(B, 0, Q, S, Q, S))
    else:
        # gamma[:, :j-2, :j-2] : B x (j-2) x (j-2) x Q x S x Q x S
        # gamma[:, 1:j-1, j-2] : B x (j-2) x Q x S x Q x S
        gamma_1 = semiring.on_tensor(gamma, lambda x: x[:, :j-2, :j-2])
        gamma_2 = semiring.on_tensor(gamma, lambda x: x[:, 1:j-1, j-2])
        return semiring.einsum(
            POP_EQUATION,
            gamma_1,
            gamma_2,
            pop,
            block_size=block_size
        )

ALPHA_EQUATION = compile_equation('biqx,biqxry->bry')

def next_alpha_column(alpha, gamma_j, j, semiring, block_size):
    # alpha : B x n x Q x S
    # gamma_j : B x j x Q x S x Q x S
    # alpha[:, :j] : B x j x Q x S
    # return : B x Q x S
    return semiring.einsum(
        ALPHA_EQUATION,
        semiring.on_tensor(alpha, lambda x: x[:, :j]),
        gamma_j,
        block_size=block_size
    )
