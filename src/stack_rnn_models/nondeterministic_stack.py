import torch
from torch_semiring_einsum import compile_equation

import attr

from lib.data_structures.linked_list import LinkedList
from lib.pytorch_tools.set_slice import set_slice
from lib.semiring import log, real, log_viterbi
from .common import StackRNNBase
from .old_nondeterministic_stack import NondeterministicStack as OldNondeterministicStack

class NondeterministicStackRNN(StackRNNBase):

    def __init__(self, input_size, num_states, stack_alphabet_size, controller,
            normalize_operations=False, include_states_in_reading=True,
            original_bottom_symbol_behavior=False, stack_reading_size=None):
        Q = num_states
        S = stack_alphabet_size
        if stack_reading_size is None:
            if include_states_in_reading:
                stack_reading_size = Q * S
            else:
                stack_reading_size = S
        super().__init__(
            input_size=input_size,
            stack_reading_size=stack_reading_size,
            controller=controller
        )
        self.num_states = num_states
        self.stack_alphabet_size = stack_alphabet_size
        self.normalize_operations = normalize_operations
        self.include_states_in_reading = include_states_in_reading
        self.original_bottom_symbol_behavior = original_bottom_symbol_behavior
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

    def forward(self, input_sequence, *args, **kwargs):
        # Automatically add the length of the input as an extra argument, which
        # allows us to allocate the proper amount of space in the
        # nondeterministic stack data structure.
        sequence_length = input_sequence.size(1)
        return super().forward(
            input_sequence,
            *args,
            sequence_length=sequence_length,
            **kwargs)

    def initial_stack(self, batch_size, reading_size, sequence_length,
            block_size, semiring=log):
        return self.get_new_stack(
            batch_size=batch_size,
            sequence_length=sequence_length,
            semiring=semiring,
            block_size=block_size
        )

    def get_new_stack(self, **kwargs):
        """Construct a new instance of the stack data structure."""
        return self.get_new_viterbi_stack(**kwargs)

    def get_new_viterbi_stack(self, batch_size, sequence_length, semiring, block_size):
        """Construct a new instance of the stack data structure, but ensure
        that it is a version that is compatible with Viterbi decoding."""
        tensor = next(self.parameters())
        if not self.original_bottom_symbol_behavior:
            return NondeterministicStack(
                batch_size=batch_size,
                num_states=self.num_states,
                stack_alphabet_size=self.stack_alphabet_size,
                sequence_length=sequence_length,
                include_states_in_reading=self.include_states_in_reading,
                block_size=block_size,
                dtype=tensor.dtype,
                device=tensor.device,
                semiring=semiring
            )
        else:
            return OldNondeterministicStack(
                batch_size=batch_size,
                num_states=self.num_states,
                stack_alphabet_size=self.stack_alphabet_size,
                sequence_length=sequence_length,
                normalize_reading=True,
                include_states_in_reading=self.include_states_in_reading,
                block_size=block_size,
                dtype=tensor.dtype,
                device=tensor.device,
                semiring=semiring
            )

    class State(StackRNNBase.State):

        def compute_stack(self, hidden_state, stack):
            push, repl, pop = actions = self.rnn.operation_log_scores(hidden_state)
            stack.update(push, repl, pop)
            return stack, actions

    def viterbi_decoder(self, input_sequence, block_size, wrapper=None):
        """Return an object that can be used to run the Viterbi algorithm on
        the stack WFA and get the best run leading up to any timestep.

        If timesteps past a certain timestep will not be used, simply slice
        the input accordingly to save computation."""
        # This allows the model to work when wrapped by RNN wrappers.
        if wrapper is not None:
            input_sequence = wrapper.wrap_input(input_sequence)
        # TODO For the limited nondeterministic stack RNN, it may be useful to
        # implement a version of this that splits the input into chunks to use
        # less memory and work on longer sequences.
        with torch.no_grad():
            outputs, operation_weights = self(
                input_sequence,
                block_size=block_size,
                return_state=False,
                include_first=False,
                return_actions=True
            )
        if operation_weights:
            # The output from the above does not include an output for timestep
            # 0, and the actions for timesteps 0 and 1 are always None, so the
            # first set of actions is expected to be None.
            assert operation_weights[0] is None
        return self.viterbi_decoder_from_operation_weights(operation_weights[1:], block_size)

    def viterbi_decoder_from_operation_weights(self, operation_weights, block_size):
        # operation_weights[0] corresponds to the action weights computed just
        # after timestep j = 1 and before j = 2.
        if not self.include_states_in_reading:
            raise NotImplementedError
        with torch.no_grad():
            batch_size = operation_weights[0][0].size(0)
            sequence_length = len(operation_weights) + 1
            # Compute the gamma and alpha tensor for every timestep in the
            # Viterbi semiring.
            stack = self.get_new_viterbi_stack(
                batch_size=batch_size,
                sequence_length=sequence_length,
                semiring=log_viterbi,
                block_size=block_size
            )
            # The first `result` returned from `update` corresponds to timestep
            # j = 1, so these lists include results starting just before
            # timestep j = 2.
            alpha_columns = []
            gamma_j_nodes = []
            alpha_j_nodes = []
            for push, repl, pop in operation_weights:
                result = stack.update(
                    log_viterbi.primitive(push),
                    log_viterbi.primitive(repl),
                    log_viterbi.primitive(pop)
                )
                # Save the nodes for the columns of gamma and alpha in lists.
                # This makes decoding simpler.
                alpha_columns.append(result.alpha_j)
                gamma_j_nodes.append(result.gamma_j[1])
                alpha_j_nodes.append(result.alpha_j[1])
        return self.get_viterbi_decoder(alpha_columns, gamma_j_nodes, alpha_j_nodes)

    def get_viterbi_decoder(self, alpha_columns, gamma_j_nodes, alpha_j_nodes):
        return ViterbiDecoder(
            alpha_columns,
            gamma_j_nodes,
            alpha_j_nodes
        )

class TooManyUpdates(ValueError):
    pass

@attr.s
class UpdateResult:
    j = attr.ib()
    gamma_j = attr.ib()
    alpha_j = attr.ib()
    gamma_prime_j = attr.ib()

class NondeterministicStack:
    """The nondeterministic stack data structure, also known as the stack WFA."""

    def __init__(self, batch_size, num_states, stack_alphabet_size,
            sequence_length, include_states_in_reading, block_size, dtype,
            device, semiring):
        """Note that `sequence_length` corresponds to the length of the input
        to the NS-RNN model. It determines the maximum number of times `update`
        can be called; `update` can be called at most `sequence_length - 1`
        times."""
        super().__init__()
        B = self.batch_size = batch_size
        Q = self.num_states = num_states
        S = self.stack_alphabet_size = stack_alphabet_size
        n = self.sequence_length = sequence_length
        self.device = device
        self.semiring = semiring
        # self.gamma[:, i+1, j, q, x, r, y] contains the value of
        # $\gamma[i \rightarrow j][q, x \rightarrow r, y]$ for 0 <= j <= n-1
        # and -1 <= i <= t-1. The value of gamma for j = n is not needed.
        # So, the size of self.gamma is n x n.
        self.gamma = semiring.zeros((B, n, n, Q, S, Q, S), dtype=dtype, device=device)
        # Initialize $\gamma[-1 \rightarrow 0]$.
        semiring.get_tensor(self.gamma)[:, gamma_i_index(-1), gamma_j_index(0), 0, 0, 0, 0] = semiring.one
        # self.alpha[:, j+1, r, y] contains the value of $\alpha[j][r, y]$
        # for -1 <= j <= n-1. The value of $\alpha[n]$ is not needed.
        # So, the size of self.alpha is n+1.
        self.alpha = semiring.zeros((B, n+1, Q, S), dtype=dtype, device=device)
        # Initialize $\alpha[-1]$ and $\alpha[0]$.
        semiring.get_tensor(self.alpha)[:, alpha_j_index(-1):alpha_j_index(0)+1, 0, 0] = semiring.one
        self.block_size = block_size
        self.include_states_in_reading = include_states_in_reading
        self.j = 0
        # self.alpha_j initially contains the value of $\alpha[0][r, y]$.
        self.alpha_j = semiring.on_tensor(self.alpha, lambda x: x[:, alpha_j_index(self.j)])

    def update(self, push, repl, pop, return_gamma_prime=False):
        # push : B x Q x S x Q x S
        # repl : B x Q x S x Q x S
        # pop : B x Q x S x Q
        if not (self.j + 1 < self.sequence_length):
            raise TooManyUpdates(
                f'attempting to compute timestep {self.j+1} (0-indexed), but '
                f'only {self.sequence_length} timesteps were allocated with '
                f'sequence_length')
        semiring = self.semiring
        block_size = self.block_size
        self.j = j = self.j + 1
        # gamma_j : B x j+1 x Q x S x Q x S
        gamma_j, gamma_prime_j = next_gamma_column(
            # B x j x j x Q x S x Q x S
            semiring.on_tensor(self.gamma, lambda x: x[:, :gamma_i_index(j-1), :gamma_j_index(j)]),
            push,
            repl,
            pop,
            semiring,
            block_size,
            return_gamma_prime
        )
        # This is just a long way of updating column j of gamma in-place.
        self.gamma = semiring.combine(
            [self.gamma, gamma_j],
            lambda args: set_slice(
                args[0],
                (slice(None), slice(None, gamma_i_index(j)), gamma_j_index(j)),
                args[1]))
        # alpha_j : B x Q x S
        self.alpha_j = next_alpha_column(
            # B x j+1 x Q x S
            semiring.on_tensor(self.alpha, lambda x: x[:, :alpha_j_index(j)]),
            # B x j+1 x Q x S x Q x S
            gamma_j,
            semiring,
            block_size
        )
        # This is just a long way of updating entry j of alpha in-place.
        self.alpha = semiring.combine(
            [self.alpha, self.alpha_j],
            lambda args: set_slice(
                args[0],
                (slice(None), alpha_j_index(j)),
                args[1]))
        return UpdateResult(j, gamma_j, self.alpha_j, gamma_prime_j)

    def reading(self):
        # Return log P_j(r, y).
        semiring = self.semiring
        # self.alpha_j : B x Q x S
        if self.include_states_in_reading:
            B = self.alpha_j.size(0)
            # result : B x (Q * S)
            result = semiring.on_tensor(self.alpha_j, lambda x: x.view(B, -1))
        else:
            # result : B x S
            result = semiring.sum(self.alpha_j, dim=1)
        if semiring is log:
            # Using softmax, normalize the log-weights so they sum to 1.
            return torch.nn.functional.softmax(result, dim=1)
        elif semiring is real:
            return result / torch.sum(result, dim=1, keepdim=True)
        else:
            raise ValueError('cannot normalize the weights in this semiring')

def ensure_not_negative(x):
    if x < 0:
        raise ValueError('index is negative')
    return x

def alpha_j_index(i):
    return ensure_not_negative(i+1)

def gamma_i_index(i):
    return ensure_not_negative(i+1)

def gamma_j_index(j):
    return ensure_not_negative(j)

REPL_EQUATION = compile_equation('biqxsz,bszry->biqxry')
GAMMA_PRIME_EQUATION = compile_equation('bktysz,bszr->bktyr')
POP_EQUATION = compile_equation('bikqxty,bktyr->biqxry')

GRAD_OF_NEG_INF_IS_ZERO = dict(grad_of_neg_inf=0.0)
NO_OPTIONS = dict()

def next_gamma_column(gamma, push, repl, pop, semiring, block_size,
        return_gamma_prime=False, gamma_prime_zero_grad=False):
    # gamma : B x T-1 x T-1 x Q x S x Q x S
    # return : B x T x Q x S x Q x S
    T = semiring.get_tensor(gamma).size(1) + 1
    B, _, _, Q, S, *_ = semiring.get_tensor(gamma).size()
    # push : B x Q x S x Q x S
    # push_term : B x 1 x Q x S x Q x S
    push_term = semiring.on_tensor(push, lambda x: x[:, None])
    # repl_term : B x T-1 x Q x S x Q x S
    if T == 1:
        repl_term = semiring.primitive(
            semiring.get_tensor(gamma).new_empty(B, 0, Q, S, Q, S))
    else:
        # Setting grad_of_neg_inf=0.0 is necessary here, because for i = -1,
        # sometimes all terms are -inf, which by default results in a gradient
        # of nan. Setting the gradient to 0 makes sense, because repl is never
        # -inf, and if gamma is all -inf, then changing repl could never cause
        # any of the terms to be greater than -inf.
        repl_term = semiring.einsum(
            REPL_EQUATION,
            # B x T-1 x Q x S x Q x S
            semiring.on_tensor(gamma, lambda x: x[:, :, -1]),
            # B x Q x S x Q x S
            repl,
            block_size=block_size,
            **(GRAD_OF_NEG_INF_IS_ZERO if semiring is log else NO_OPTIONS)
        )
    # pop_term : B x T-2 x Q x S x Q x S
    if T <= 2:
        gamma_prime = None
        pop_term = semiring.primitive(
            semiring.get_tensor(gamma).new_empty(B, 0, Q, S, Q, S))
    else:
        # gamma_prime : B x T-2 x Q x S x Q
        gamma_prime = semiring.einsum(
            GAMMA_PRIME_EQUATION,
            # B x T-2 x Q x S x Q x S
            semiring.on_tensor(gamma, lambda x: x[:, 1:, -1]),
            # B x Q x S x Q
            pop,
            block_size=block_size,
            **(GRAD_OF_NEG_INF_IS_ZERO if gamma_prime_zero_grad and semiring is log else NO_OPTIONS)
        )
        # See note about grad_of_neg_inf above.
        pop_term = semiring.einsum(
            POP_EQUATION,
            # B x T-2 x T-2 x Q x S x Q x S
            semiring.on_tensor(gamma, lambda x: x[:, :-1, :-1]),
            # B x Q x S x Q
            gamma_prime,
            block_size=block_size,
            **(GRAD_OF_NEG_INF_IS_ZERO if semiring is log else NO_OPTIONS)
        )
        if not return_gamma_prime:
            gamma_prime = None
    gamma_j = semiring.combine([
        semiring.add(
            semiring.on_tensor(repl_term, lambda x: x[:, :-1]),
            pop_term
        ),
        semiring.on_tensor(repl_term, lambda x: x[:, -1:]),
        push_term
    ], lambda args: torch.cat(args, dim=1))
    return gamma_j, gamma_prime

ALPHA_EQUATION = compile_equation('biqx,biqxry->bry')

def next_alpha_column(alpha, gamma_j, semiring, block_size):
    # alpha : B x T x Q x S
    # gamma_j : B x T x Q x S x Q x S
    # return : B x Q x S
    return semiring.einsum(
        ALPHA_EQUATION,
        alpha,
        gamma_j,
        block_size=block_size
    )

@attr.s(frozen=True)
class Operation:
    state_to = attr.ib()

@attr.s(frozen=True)
class PushOperation(Operation):
    symbol = attr.ib()

@attr.s(frozen=True)
class ReplaceOperation(Operation):
    symbol = attr.ib()

@attr.s(frozen=True)
class PopOperation(Operation):
    pass

class ViterbiDecoder:

    def __init__(self, alpha_columns, gamma_j_nodes, alpha_j_nodes):
        self.alpha_columns = alpha_columns
        self.gamma_j_nodes = gamma_j_nodes
        self.alpha_j_nodes = alpha_j_nodes

    def decode_timestep(self, j):
        """For each batch element, return the highest-weighted PDA run leading
        up to the prediction at timestep j, as well as its score. Let n be the
        length of the input sequence. Timesteps are 0-indexed, where j = 0
        corresponds to the first input symbol, and j = n-1 is the last valid
        timestep, corresponding to the last input symbol.

        The Viterbi path leading up to timestep j is always of length j,
        because every run starts just after timestep j = 0, after the first
        input symbol has been read. So, every run leading up to j = 0 is empty,
        and the shortest non-empty runs end at j = 1.

        Since there is nothing to decode for j = 0, it is not allowed; it
        would just be an empty sequence of actions anyway.

        There is nothing to decode for timestep n because the prediction for
        the symbol at timestep n (e.g. EOS) is computed directly from the
        hidden state for n-1, so no stack actions are needed after n-1."""
        if not (1 <= j <= self.sequence_length - 1):
            raise ValueError(f'timestep ({j}) must be in the range [1, {self.sequence_length-1}]')
        # Sum over states, then stack symbols.
        alpha_j_sum_scores, alpha_j_sum_node = \
            log_viterbi.sum(log_viterbi.sum(self.get_alpha_j(j), dim=1), dim=1)
        batch_size = alpha_j_sum_scores.size(0)
        paths = [
            self.decode_alpha_j_sum(alpha_j_sum_node, b, j)
            for b in range(batch_size)
        ]
        return paths, alpha_j_sum_scores

    def decode_alpha_j_sum(self, alpha_j_sum_node, b, j):
        y = alpha_j_sum_node.backpointers[b]
        alpha_j_sum_states_node, = alpha_j_sum_node.children
        r = alpha_j_sum_states_node.backpointers[b, y]
        return self.decode_alpha_j(b, j, r, y)

    def decode_alpha_j(self, b, j, r, y):
        if j > 0:
            alpha_j_node = self.get_alpha_j_node(j)
            i_alpha_index, q, x = alpha_j_node.backpointers[b, r, y]
            i = self.i_from_alpha_index(i_alpha_index)
            # Recurse on alpha[i] and gamma[i, j]
            alpha_path = self.decode_alpha_j(b, i, q, x)
            gamma_path = self.decode_gamma_j(b, i, j, q, x, r, y)
            path = alpha_path
            path.extend(gamma_path)
            return path
        elif -1 <= j <= 0:
            # The first valid timestep for alpha is -1; both timesteps 0 and -1
            # are initialized to be in state q0 and have the bottom symbol on
            # the stack, where the step from -1 to 0 represents a fake "push"
            # of the bottom symbol to the stack (but this "push" should not be
            # included in the Viterbi path). In both cases, we should return an
            # empty list of operations.
            return LinkedList([])
        else:
            raise ValueError(f'logic error: invalid value for j ({j})')

    def decode_gamma_j(self, b, i, j, q, x, r, y):
        if j == 0:
            # For timestep 0, return an empty list of operations. See note in
            # decode_alpha_j.
            return LinkedList([])
        else:
            gamma_j_node = self.get_gamma_j_node(j)
            repl_pop_node, repl_node, push_node = gamma_j_node.children
            if i < j-2:
                is_pop = repl_pop_node.backpointers[b, self.i_to_gamma_index(i), q, x, r, y].item()
                repl_node, pop_node = repl_pop_node.children
                if is_pop:
                    return self.decode_pop(pop_node, b, i, j, q, x, r, y)
                else:
                    return self.decode_repl(repl_node, b, i, j, q, x, r, y)
            elif i == j-2:
                return self.decode_repl(repl_node, b, i, j, q, x, r, y)
            elif i == j-1:
                return LinkedList([PushOperation(r.item(), y.item())])
            else:
                raise ValueError

    def decode_repl(self, repl_node, b, i, j, q, x, r, y):
        s, z = repl_node.backpointers[b, self.i_to_gamma_index(i), q, x, r, y]
        path = self.decode_gamma_j(b, i, j-1, q, x, s, z)
        path.append(ReplaceOperation(r.item(), y.item()))
        return path

    def decode_pop(self, pop_node, b, i, j, q, x, r, y):
        k_pop_index, t = pop_node.backpointers[b, self.i_to_gamma_index(i), q, x, r, y]
        gamma_1_node, gamma_prime_node = pop_node.children
        s, z = gamma_prime_node.backpointers[b, k_pop_index, t, y, r]
        k = self.k_from_pop_index(k_pop_index)
        gamma_1_path = self.decode_gamma_j(b, i, k, q, x, t, y)
        gamma_2_path = self.decode_gamma_j(b, k, j-1, t, y, s, z)
        path = gamma_1_path
        path.extend(gamma_2_path)
        path.append(PopOperation(r.item()))
        return path

    def get_alpha_j(self, j):
        # self.alpha_columns[0] is actually alpha[1], so we need to adjust the
        # index accordingly.
        return self.alpha_columns[ensure_not_negative(j-1)]

    def get_alpha_j_node(self, j):
        return self.alpha_j_nodes[ensure_not_negative(j-1)]

    def get_gamma_j_node(self, j):
        # Return the node for computing all gamma entries of the form
        # gamma[i, j].
        return self.gamma_j_nodes[ensure_not_negative(j-1)]

    def i_from_alpha_index(self, index):
        # The first index of alpha corresponds to i = -1, so subtract 1.
        return index - 1

    def i_to_gamma_index(self, i):
        return gamma_i_index(i)

    def k_from_pop_index(self, index):
        # The first row of gamma corresponds to i = -1, and the einsum for the
        # pop rule starts at gamma[:, 1], so an index of 0 in the einsum
        # corresponds to k = 0.
        return index

    @property
    def sequence_length(self):
        return len(self.alpha_columns) + 1
