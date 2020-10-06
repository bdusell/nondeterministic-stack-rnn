import torch
from torch_semiring_einsum import compile_equation

from ..pytorch_tools.set_slice import set_slice
from ..pytorch_tools.unidirectional_rnn import UnidirectionalLSTM
from ..semiring import log
from .common import StackRNNBase

class NondeterministicStackRNN(StackRNNBase):

    def __init__(self, input_size, hidden_units, num_states,
            stack_alphabet_size, controller=UnidirectionalLSTM):
        super().__init__(
            input_size, hidden_units, stack_alphabet_size, controller)
        self.num_states = num_states
        self.stack_alphabet_size = stack_alphabet_size
        Q = num_states
        S = stack_alphabet_size
        self.num_op_rows = Q * S
        self.num_op_cols = Q * S + Q * S + Q
        self.operation_layer = torch.nn.Linear(
            hidden_units,
            self.num_op_rows * self.num_op_cols
        )

    def operation_log_probs(self, hidden_state):
        B = hidden_state.size(0)
        Q = self.num_states
        S = self.stack_alphabet_size
        # logits : B x (Q * S) x (Q * S + Q * S + Q)
        logits = self.operation_layer(hidden_state)
        # logits_view : B x (Q * S) x (Q * S + Q * S + Q)
        logits_view = logits.view(B, self.num_op_rows, self.num_op_cols)
        # log_probs : B x (Q * S) x (Q * S + Q * S + Q)
        log_probs = torch.nn.functional.log_softmax(logits_view, dim=2)
        push_chunk, repl_chunk, pop_chunk = log_probs.split([Q * S, Q * S, Q], dim=2)
        push = push_chunk.view(B, Q, S, Q, S)
        repl = repl_chunk.view(B, Q, S, Q, S)
        pop = pop_chunk.view(B, Q, S, Q)
        return push, repl, pop

    def generate_outputs(self, input_tensors, *args, **kwargs):
        # Automatically use the sequence length to determine the size of the
        # alpha and gamma tensors.
        return super().generate_outputs(
            input_tensors,
            sequence_length=input_tensors.size(0),
            *args,
            **kwargs)

    def initial_stack(self, batch_size, stack_size, device, sequence_length,
            block_size):
        tensor = next(self.parameters())
        return NondeterministicStack(
            batch_size,
            self.num_states,
            self.stack_alphabet_size,
            sequence_length,
            block_size,
            tensor.dtype,
            device
        )

    class State(StackRNNBase.State):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.block_size = None

        def compute_stack(self, hidden_state, stack):
            push, repl, pop = self.rnn.operation_log_probs(hidden_state)
            stack.update(push, repl, pop)
            return stack

class NondeterministicStack:

    def __init__(self, B, Q, S, n, block_size, dtype, device):
        super().__init__()
        self.semiring = semiring = log
        self.alpha = semiring.zeros((B, n, Q, S), dtype=dtype, device=device)
        self.gamma = semiring.zeros((B, n-1, n-1, Q, S, Q, S), dtype=dtype, device=device)
        semiring.get_tensor(self.alpha)[:, 0, 0, 0] = semiring.one
        self.block_size = block_size
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
        # Return log P_j(y).
        # alpha[0...j] has already been computed.
        semiring = self.semiring
        # alpha_j : B x Q x S
        alpha_j = self.alpha[:, self.j]
        # logits : B x S
        logits = semiring.sum(alpha_j, dim=1)
        # Using softmax, normalize the log-weights so they sum to 1.
        assert semiring is log
        # return : B x S
        return torch.nn.functional.softmax(logits, dim=1)

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
