import torch
from torch_semiring_einsum import compile_equation

from lib.pytorch_tools.set_slice import set_slice
from lib.semiring import log
from .common import StackRNNBase
from .nondeterministic_stack import (
    NondeterministicStackRNN,
    NondeterministicStack,
    gamma_i_index,
    gamma_j_index,
    alpha_j_index
)
from .old_vector_nondeterministic_stack import (
    VectorNondeterministicStack as OldVectorNondeterministicStack
)

zeta_i_index = gamma_i_index
zeta_j_index = gamma_j_index

class VectorNondeterministicStackRNN(NondeterministicStackRNN):

    def __init__(self, input_size, num_states, stack_alphabet_size,
            stack_embedding_size, controller, normalize_operations=False,
            include_states_in_reading=True,
            original_bottom_symbol_behavior=False,
            bottom_vector='learned',
            **kwargs):
        Q = num_states
        S = stack_alphabet_size
        m = stack_embedding_size
        if not include_states_in_reading:
            raise ValueError('include_states_in_reading=False is not supported')
        super().__init__(
            input_size=input_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            controller=controller,
            normalize_operations=normalize_operations,
            original_bottom_symbol_behavior=original_bottom_symbol_behavior,
            stack_reading_size=Q * S * m,
            include_states_in_reading=include_states_in_reading,
            **kwargs
        )
        self.stack_embedding_size = stack_embedding_size
        self.pushed_vector_layer = torch.nn.Sequential(
            torch.nn.Linear(
                self.controller.output_size(),
                stack_embedding_size
            ),
            torch.nn.LogSigmoid()
        )
        # This parameter is the learned embedding that always sits at the
        # bottom of the stack. It is the input to a sigmoid operation, so the
        # vector used in the stack will be in (0, 1).
        if original_bottom_symbol_behavior:
            if bottom_vector is not None:
                raise ValueError(
                    'if original_bottom_symbol_behavior=True, bottom_vector '
                    'must be None')
        else:
            self.bottom_vector_type = bottom_vector
            if bottom_vector == 'learned':
                self.bottom_vector = torch.nn.Parameter(torch.zeros((m,)))
            elif bottom_vector in ('one', 'zero'):
                pass
            else:
                raise ValueError(f'unknown bottom vector option: {bottom_vector!r}')

    def pushed_vector(self, hidden_state):
        return self.pushed_vector_layer(hidden_state)

    def get_bottom_vector(self, semiring):
        if self.bottom_vector_type == 'learned':
            if semiring is not log:
                raise NotImplementedError
            return torch.nn.functional.logsigmoid(self.bottom_vector)
        elif self.bottom_vector_type == 'one':
            tensor = next(self.parameters())
            return semiring.ones((self.stack_embedding_size,), like=tensor)
        elif self.bottom_vector_type == 'zero':
            tensor = next(self.parameters())
            return semiring.zeros((self.stack_embedding_size,), like=tensor)
        else:
            raise ValueError

    def get_new_stack(self, batch_size, sequence_length, semiring, block_size):
        tensor = next(self.parameters())
        if not self.original_bottom_symbol_behavior:
            return VectorNondeterministicStack(
                batch_size=batch_size,
                num_states=self.num_states,
                stack_alphabet_size=self.stack_alphabet_size,
                stack_embedding_size=self.stack_embedding_size,
                sequence_length=sequence_length,
                bottom_vector=self.get_bottom_vector(semiring),
                block_size=block_size,
                dtype=tensor.dtype,
                device=tensor.device,
                semiring=semiring
            )
        else:
            return OldVectorNondeterministicStack(
                batch_size=batch_size,
                num_states=self.num_states,
                stack_alphabet_size=self.stack_alphabet_size,
                stack_embedding_size=self.stack_embedding_size,
                sequence_length=sequence_length,
                block_size=block_size,
                dtype=tensor.dtype,
                device=tensor.device,
                semiring=semiring
            )

    class State(StackRNNBase.State):

        def compute_stack(self, hidden_state, stack):
            push, repl, pop = self.rnn.operation_log_scores(hidden_state)
            pushed_vector = self.rnn.pushed_vector(hidden_state)
            actions = (push, repl, pop, pushed_vector)
            stack.update(push, repl, pop, pushed_vector)
            return stack, actions

class VectorNondeterministicStack(NondeterministicStack):

    def __init__(self, batch_size, num_states, stack_alphabet_size,
            stack_embedding_size, sequence_length, bottom_vector, block_size,
            dtype, device, semiring):
        super().__init__(
            batch_size=batch_size,
            num_states=num_states,
            stack_alphabet_size=stack_alphabet_size,
            sequence_length=sequence_length,
            include_states_in_reading=True,
            block_size=block_size,
            dtype=dtype,
            device=device,
            semiring=semiring
        )
        B = self.batch_size
        Q = self.num_states
        S = self.stack_alphabet_size
        m = self.stack_embedding_size = stack_embedding_size
        n = self.sequence_length
        # self.zeta[:, i+1, j, q, x, r, y] contains the value of
        # $\zeta[i \rightarrow j][q, x \rightarrow r, y]$ for 0 <= j <= n-1
        # and -1 <= i <= t-1. The value of zeta for j = n is not needed.
        # So, the size of self.zeta is n x n.
        self.zeta = semiring.zeros((B, n, n, Q, S, Q, S, m), dtype=dtype, device=device)
        # Initialize $\zeta[-1 \rightaarrow 0]$ to the (possibly learned)
        # bottom vector.
        self.zeta = semiring.combine(
            [self.zeta, bottom_vector],
            lambda args: set_slice(
                args[0],
                (slice(None), zeta_i_index(-1), zeta_j_index(0)),
                args[1]))
        self.zeta_j = semiring.on_tensor(
            self.zeta,
            lambda x: x[:, :zeta_i_index(self.j), zeta_j_index(self.j)])

    def update(self, push, repl, pop, pushed_vector):
        # push : B x Q x S x Q x S
        # repl : B x Q x S x Q x S
        # pop : B x Q x S x Q
        # pushed_vector : B x m
        # Update the self.gamma and self.alpha tables.
        result = super().update(push, repl, pop, return_gamma_prime=True)
        semiring = self.semiring
        block_size = self.block_size
        j = self.j
        # self.zeta_j : B x j+1 x Q x S x Q x S x m
        self.zeta_j = next_zeta_column(
            # B x j x j x Q x S x Q x S x m
            semiring.on_tensor(self.zeta, lambda x: x[:, :zeta_i_index(j-1), :zeta_j_index(j)]),
            # B x j-1 x Q x S x Q
            result.gamma_prime_j,
            push,
            repl,
            pushed_vector,
            semiring,
            block_size
        )
        result.gamma_prime_j = None
        self.zeta = semiring.combine(
            [self.zeta, self.zeta_j],
            lambda args: set_slice(
                args[0],
                (slice(None), slice(None, zeta_i_index(j)), zeta_j_index(j)),
                args[1]))
        return result

    def reading(self):
        semiring = self.semiring
        # eta_j : B x Q x S x m
        eta_j = next_eta_column(
            semiring.on_tensor(self.alpha, lambda x: x[:, :alpha_j_index(self.j)]),
            self.zeta_j,
            semiring,
            self.block_size
        )
        return eta_to_reading(self.alpha_j, eta_j, semiring)

ZETA_REPL_EQUATION = compile_equation('biqxszm,bszry->biqxrym')
ZETA_POP_EQUATION = compile_equation('bikqxtym,bktyr->biqxrym')

def next_zeta_column(zeta, gamma_prime_j, push, repl, pushed_vector, semiring,
        block_size):
    # zeta : B x T-1 x T-1 x Q x S x Q x S x m
    # gamma_prime_j : B x T-2 x Q x S x Q
    # return : B x T x Q x S x Q x S x m
    T = semiring.get_tensor(zeta).size(1) + 1
    B, _, _, Q, S, _, _, m = semiring.get_tensor(zeta).size()
    # push : B x Q x S x Q x S
    # pushed_vector : B x m
    # push_term : B x 1 x Q x S x Q x S x m
    push_term = semiring.on_tensor(
        # B x Q x S x Q x S x m
        semiring.multiply(
            # B x Q x S x Q x S x 1
            semiring.on_tensor(push, lambda x: x[:, :, :, :, :, None]),
            # B x 1 x 1 x 1 x 1 x m
            semiring.on_tensor(pushed_vector, lambda x: x[:, None, None, None, None, :])
        ),
        lambda x: x[:, None]
    )
    # repl_term : B x T-1 x Q x S x Q x S x m
    if T == 1:
        repl_term = semiring.primitive(
            semiring.get_tensor(zeta).new_empty(B, 0, Q, S, Q, S, m))
    else:
        repl_term = semiring.einsum(
            ZETA_REPL_EQUATION,
            # B x T-1 x Q x S x Q x S x m
            semiring.on_tensor(zeta, lambda x: x[:, :, -1]),
            # B x Q x S x Q x S
            repl,
            block_size=block_size,
            **(dict(grad_of_neg_inf=0.0) if semiring is log else {})
        )
    # pop_term : B x T-2 x Q x S x Q x S x m
    if T <= 2:
        pop_term = semiring.primitive(
            semiring.get_tensor(zeta).new_empty(B, 0, Q, S, Q, S, m))
    else:
        pop_term = semiring.einsum(
            ZETA_POP_EQUATION,
            # B x T-2 x T-2 x Q x S x Q x S x m
            semiring.on_tensor(zeta, lambda x: x[:, :-1, :-1]),
            # B x T-2 x Q x S x Q
            gamma_prime_j,
            block_size=block_size,
            **(dict(grad_of_neg_inf=0.0) if semiring is log else {})
        )
    return semiring.combine([
        semiring.add(
            semiring.on_tensor(repl_term, lambda x: x[:, :-1]),
            pop_term
        ),
        semiring.on_tensor(repl_term, lambda x: x[:, -1:]),
        push_term
    ], lambda args: torch.cat(args, dim=1))

ETA_EQUATION = compile_equation('biqx,biqxrym->brym')

def next_eta_column(alpha, zeta_j, semiring, block_size):
    # alpha : B x T x Q x S
    # zeta_j : B x T x Q x S x Q x S x m
    # return : B x Q x S x m
    return semiring.einsum(
        ETA_EQUATION,
        alpha,
        zeta_j,
        block_size=block_size
    )

def eta_to_reading(alpha_j, eta_j, semiring):
    assert semiring is log
    # alpha_j : B x Q x S
    # eta_j : B x Q x S x m
    # denom : B
    denom = semiring.sum(alpha_j, dim=(1, 2))
    # Divide (in log space) eta by the sum over alpha, then take the exp
    # to get back to real space. Finally, flatten the dimensions.
    B = eta_j.size(0)
    return torch.exp(eta_j - denom[:, None, None, None]).view(B, -1)
