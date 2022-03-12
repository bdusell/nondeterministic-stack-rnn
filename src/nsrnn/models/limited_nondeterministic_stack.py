import attr
import torch
from torch_semiring_einsum import compile_equation

from torch_rnn_tools import UnidirectionalRNN
from ..pytorch_tools.set_slice import set_slice
from ..semiring import log, log_viterbi
from ..data_structures.linked_list import LinkedList
from .nondeterministic_stack import NondeterministicStackRNN

class LimitedNondeterministicStackRNN(NondeterministicStackRNN):

    def __init__(self, input_size, num_states, stack_alphabet_size,
            window_size, controller, normalize_operations=True,
            normalize_reading=True, include_states_in_reading=True):
        super().__init__(
            input_size, num_states, stack_alphabet_size, controller,
            normalize_operations, normalize_reading, include_states_in_reading)
        self.window_size = window_size

    # Use a separate type for the initial state used with `initial_state()`,
    # `initial_state=` and `return_state=` that contains only the chunks of
    # gamma and alpha that need to be passed between batches.
    @attr.s
    class InitialState(UnidirectionalRNN.State):

        state = attr.ib()
        stack = attr.ib()

        def detach(self):
            return type(self)(self.state.detach(), self.stack.detach())

        def batch_size(self):
            return self.state.batch_size()

        def slice_batch(self, s):
            return type(self)(self.state.slice_batch(s), self.stack.slice_batch(s))

    class InitialStackState(UnidirectionalRNN.State):

        def __init__(self, gamma, alpha, semiring):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.semiring = semiring

        def transform_tensors(self, func):
            # This implements detach() and slice_batch().
            return type(self)(
                self.semiring.on_tensor(self.gamma, func),
                self.semiring.on_tensor(self.alpha, func),
                self.semiring
            )

        def batch_size(self):
            return self.gamma.size(0)

    def forward(self, input_sequence, block_size, initial_state=None,
            return_state=False, include_first=True, return_signals=False):
        semiring = log
        batch_size = input_sequence.size(0)
        sequence_length = input_sequence.size(1)
        if initial_state is None:
            initial_state = self.initial_state(batch_size, semiring)
        else:
            if not isinstance(initial_state, self.InitialState):
                raise TypeError
        # Convert initial_state from the InitialState type to State.
        tensor = next(self.parameters())
        stack = LimitedNondeterministicStack(
            num_states=self.num_states,
            stack_alphabet_size=self.stack_alphabet_size,
            # The last time step can be skipped unless returning the last
            # state or returning operation weights.
            sequence_length=sequence_length - int(not (return_state or return_signals)),
            window_size=self.window_size,
            normalize_reading=self.normalize_reading,
            include_states_in_reading=self.include_states_in_reading,
            initial_state=initial_state.stack,
            semiring=semiring,
            block_size=block_size,
            dtype=tensor.dtype,
            device=tensor.device
        )
        state = self.State(
            rnn=self,
            state=initial_state.state,
            stack=stack,
            previous_stack=None,
            return_signals=return_signals
        )
        result = super(NondeterministicStackRNN, self).forward(
            input_sequence,
            initial_state=state,
            return_state=return_state,
            include_first=include_first,
            return_signals=return_signals
        )
        if return_state:
            output, last_state = result
            # Make sure that the stack for the last time step has been
            # computed, even though it is not required for the last output.
            last_state.get_stack()
            # Convert the last state from the State type to InitialState.
            D = self.window_size
            last_state = self.InitialState(
                last_state.state,
                self.InitialStackState(
                    gamma=semiring.on_tensor(last_state.stack.gamma, lambda x: x[:, -(D-1):, -(D-1):]),
                    alpha=semiring.on_tensor(last_state.stack.alpha, lambda x: x[:, -D:]),
                    semiring=semiring
                )
            )
            result = output, last_state
        return result

    def viterbi_decoder(self, input_sequence, block_size, wrapper=None):
        """Return an object that can be used to run the Viterbi algorithm on
        the stack WFA and get the best run leading up to any timestep.

        If timesteps past a certain timestep will not be used, simply slice
        the input accordingly to save computation."""
        # This allows the model to work when wrapped by RNN wrappers.
        if wrapper is not None:
            input_sequence = wrapper.wrap_input(input_sequence)
        # TODO It may be useful to implement a version of this that splits the
        # input into chunks to use less memory and work on longer sequences.
        with torch.no_grad():
            outputs, operation_weights = self(
                input_sequence,
                block_size=block_size,
                return_state=False,
                include_first=False,
                return_signals=True
            )
        return self.viterbi_decoder_from_operation_weights(operation_weights, block_size)

    def viterbi_decoder_from_operation_weights(self, operation_weights, block_size):
        with torch.no_grad():
            tensor = next(self.parameters())
            batch_size = operation_weights[0][0].size(0)
            sequence_length = len(operation_weights)
            # Compute the gamma and alpha tensor for every timestep in the
            # Viterbi semiring.
            initial_stack_state = self.initial_stack_state(
                batch_size=batch_size,
                semiring=log_viterbi
            )
            stack = LimitedNondeterministicStack(
                num_states=self.num_states,
                stack_alphabet_size=self.stack_alphabet_size,
                sequence_length=sequence_length,
                window_size=self.window_size,
                normalize_reading=self.normalize_reading,
                include_states_in_reading=self.include_states_in_reading,
                initial_state=initial_stack_state,
                semiring=log_viterbi,
                block_size=block_size,
                dtype=tensor.dtype,
                device=tensor.device
            )
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
        return ViterbiDecoder(alpha_columns, gamma_j_nodes, alpha_j_nodes, self.window_size)

    def initial_state(self, batch_size, semiring=log):
        return self.InitialState(
            self.controller.initial_state(batch_size),
            self.initial_stack_state(batch_size, semiring)
        )

    def initial_stack_state(self, batch_size, semiring):
        # Return dummy chunks of gamma and alpha set to zeros.
        device = self.device
        B = batch_size
        D = self.window_size
        Q = self.num_states
        S = self.stack_alphabet_size
        # The first time step j to be computed is 1 (gamma[0, 0] = 0 anyway).
        # So, the first chunk of gamma corresponds to gamma[-D+1:1, -D+1:1] and
        # is all set to 0.
        # The first chunk of alpha corresponds to alpha[-D+1:1], where
        # alpha[-D+1:0] is set to 0 and alpha[0] is set to 1 for the initial
        # PDA state and bottom symbol.
        # NOTE The initial gamma chunk should be set to 0 (-inf in log space),
        # but this causes the gradient to be nan. To avoid this, we instead set
        # it to 1. This is okay because it gets multiplied by alpha values that
        # are 0 anyway, so the value doesn't matter and gives the same
        # mathematical result no matter what.
        gamma = semiring.ones((B, D-1, D-1, Q, S, Q, S), device=device)
        alpha = semiring.zeros((B, D, Q, S), device=device)
        semiring.get_tensor(alpha)[:, -1, 0, 0] = semiring.one
        return self.InitialStackState(gamma, alpha, semiring)

class LimitedNondeterministicStack:

    def __init__(self, num_states, stack_alphabet_size, sequence_length,
            window_size, normalize_reading, include_states_in_reading,
            initial_state, semiring, block_size, dtype, device):
        # Let j_1 be the first time step for which we will be computing gamma
        # and alpha, that is, gamma[i, j_1] and alpha[j_1].
        # We will be computing gamma and alpha for j = j_1 through j_1 + T - 1.
        # Since i >= j - D and i <= j - 1, computing gamma[i, j_1] only
        # requires computing gamma for i = j_1 - D through j_1 - 1.
        # Computing gamma[i -> j_1] for i = j_1 - D through j_1 - 1 only
        # requires the values of gamma[i, j] for j_1 - D + 1 <= j <= j_1 - 1.
        # Computing alpha[j_1] only requires alpha[j] for j = j_1 - D through
        # j_1 - 1 and gamma[i -> j_1] for i = j_1 - D through j_1 - 1.
        # This means that the initial stack state consists of:
        # (1) a square of gamma for gamma[j_1 - D + 1:j_1, j_1 - D + 1:j_1]
        # (2) a slice of alpha for alpha[j_1 - D:j_1].
        super().__init__()
        B = semiring.get_tensor(initial_state.gamma).size(0)
        Q = num_states
        S = stack_alphabet_size
        T = sequence_length
        D = window_size
        if semiring.get_tensor(initial_state.gamma).size() != (B, D-1, D-1, Q, S, Q, S):
            raise ValueError
        if semiring.get_tensor(initial_state.alpha).size() != (B, D, Q, S):
            raise ValueError
        # Initialize the alpha tensor with zeros and the piece passed in from
        # the previous batch.
        self.alpha = semiring.zeros((B, D+T, Q, S), dtype=dtype, device=device)
        if not normalize_reading:
            # If the stack reading is not going to be normalized, do not use
            # -inf for the 0 weights in the initial time step, but use a
            # really negative number. This avoids nans.
            semiring.get_tensor(self.alpha)[:, 0, :, :] = -1e10
        semiring.get_tensor(self.alpha)[:, :D] = semiring.get_tensor(initial_state.alpha)
        # Initialize the gamma tensor with zeros and the corner passed in from
        # the previous batch.
        self.gamma = semiring.zeros((B, T+D-1, T+D-1, Q, S, Q, S), dtype=dtype, device=device)
        semiring.get_tensor(self.gamma)[:, :D-1, :D-1] = semiring.get_tensor(initial_state.gamma)
        self.window_size = D
        self.semiring = semiring
        self.block_size = block_size
        self.normalize_reading = normalize_reading
        self.include_states_in_reading = include_states_in_reading
        self.j = 0

    @attr.s
    class UpdateResult:
        j = attr.ib()
        gamma_j = attr.ib()
        alpha_j = attr.ib()

    def update(self, push, repl, pop):
        # push : B x Q x S x Q x S
        # repl : B x Q x S x Q x S
        # pop : B x Q x S x Q
        semiring = self.semiring
        D = self.window_size
        block_size = self.block_size
        j = self.j
        # gamma_j : B x D x Q x S x Q x S x D
        gamma_j = next_gamma_column(
            semiring.on_tensor(self.gamma, lambda x: x[:, j:j+D-1, j:j+(D-1)]),
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
                (slice(None), slice(j, j+D), D-1+j),
                args[1]))
        alpha_j = next_alpha_column(
            semiring.on_tensor(self.alpha, lambda x: x[:, j:j+D]),
            gamma_j,
            semiring,
            block_size
        )
        self.alpha = semiring.combine(
            [self.alpha, alpha_j],
            lambda args: set_slice(
                args[0],
                (slice(None), D+j),
                args[1]))
        self.j += 1
        return self.UpdateResult(j, gamma_j, alpha_j)

    def reading(self):
        semiring = self.semiring
        # Return log P_j(r, y).
        # alpha[0...j] has already been computed.
        D = self.window_size
        # alpha_j : B x Q x S
        alpha_j = self.alpha[:, D + self.j - 1]
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

REPL_EQUATION = compile_equation('biqxsz,bszry->biqxry')
POP_EQUATION = compile_equation('bikqxty,bktysz,bszr->biqxry')

def next_gamma_column(gamma, push, repl, pop, semiring, block_size):
    # gamma : B x D-1 x D-1 x Q x S x Q x S
    # return : B x D x Q x S x Q x S
    D = semiring.get_tensor(gamma).size(1) + 1
    # push : B x Q x S x Q x S
    # push_term : B x 1 x Q x S x Q x S
    push_term = semiring.on_tensor(push, lambda x: x[:, None])
    # gamma[:, :, -1] : B x D-1 x Q x S x Q x S
    # repl : B x Q x S x Q x S
    # repl_term : B x D-1 x Q x S x Q x S
    repl_term = semiring.einsum(
        REPL_EQUATION,
        semiring.on_tensor(gamma, lambda x: x[:, :, -1]),
        repl,
        block_size=block_size
    )
    # gamma[:, :-1, :-1] : B x D-2 x D-2 x Q x S x Q x S
    # gamma[:, 1:, -1] : B x D-2 x Q x S x Q x S
    # pop : B x Q x S x Q
    # pop_term : B x D-2 x Q x S x Q x S
    pop_term = semiring.einsum(
        POP_EQUATION,
        semiring.on_tensor(gamma, lambda x: x[:, :-1, :-1]),
        semiring.on_tensor(gamma, lambda x: x[:, 1:, -1]),
        pop,
        block_size=block_size
    )
    return semiring.combine([
        semiring.add(
            semiring.on_tensor(repl_term, lambda x: x[:, :-1]),
            pop_term
        ),
        semiring.on_tensor(repl_term, lambda x: x[:, -1:]),
        push_term
    ], lambda args: torch.cat(args, dim=1))

ALPHA_EQUATION = compile_equation('biqx,biqxry->bry')

def next_alpha_column(alpha, gamma_j, semiring, block_size):
    # alpha : B x D x Q x S
    # gamma_j : B x D x Q x S x Q x S
    # return : B x Q x S
    return semiring.einsum(
        ALPHA_EQUATION,
        alpha,
        gamma_j,
        block_size=block_size
    )

class ViterbiDecoder:

    def __init__(self, alpha_columns, gamma_j_nodes, alpha_j_nodes, window_size):
        self.alpha_columns = alpha_columns
        self.gamma_j_nodes = gamma_j_nodes
        self.alpha_j_nodes = alpha_j_nodes
        self.window_size = window_size

    def decode_timestep(self, j):
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
            relative_i, q, x = alpha_j_node.backpointers[b, r, y]
            i = j - self.window_size + relative_i.item()
            # Recurse on alpha[i] and gamma[i, j]
            alpha_path = self.decode_alpha_j(b, i, q, x)
            gamma_path = self.decode_gamma_j(b, i, j, q, x, r, y)
            path = alpha_path
            path.extend(gamma_path)
            return path
        elif j == 0:
            return LinkedList([])
        else:
            raise ValueError

    def decode_gamma_j(self, b, i, j, q, x, r, y):
        gamma_j_node = self.get_gamma_j_node(j)
        repl_pop_node, repl_node, push_node = gamma_j_node.children
        if i < j - self.window_size:
            raise ValueError
        elif i < j-2:
            relative_i = self.get_relative_i(i, j)
            is_pop = repl_pop_node.backpointers[b, relative_i, q, x, r, y].item()
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
        relative_i = self.get_relative_i(i, j)
        s, z = repl_node.backpointers[b, relative_i, q, x, r, y]
        path = self.decode_gamma_j(b, i, j-1, q, x, s, z)
        path.append(ReplaceOperation(r.item(), y.item()))
        return path

    def decode_pop(self, pop_node, b, i, j, q, x, r, y):
        relative_i = self.get_relative_i(i, j)
        relative_k, t, s, z = pop_node.backpointers[b, relative_i, q, x, r, y]
        k = j - (self.window_size - 1) + relative_k.item()
        gamma_1_path = self.decode_gamma_j(b, i, k, q, x, t, y)
        gamma_2_path = self.decode_gamma_j(b, k, j-1, t, y, s, z)
        path = gamma_1_path
        path.extend(gamma_2_path)
        path.append(PopOperation(r.item()))
        return path

    def get_alpha_j(self, j):
        # Let 0 be the first time step, where alpha[0] is the set of initial
        # weights where only alpha[0][0, 0] is set to 1.
        # self.alpha_columns[0] is actually alpha[1], so we need to adjust the
        # index accordingly.
        return self.alpha_columns[j-1]

    def get_alpha_j_node(self, j):
        return self.alpha_j_nodes[j-1]

    def get_gamma_j_node(self, j):
        # Return the node for computing all gamma entries of the form
        # gamma[i, j].
        return self.gamma_j_nodes[j-1]

    def get_relative_i(self, i, j):
        return i - (j - self.window_size)

@attr.s
class Operation:
    state_to = attr.ib()

@attr.s
class PushOperation(Operation):
    symbol = attr.ib()

@attr.s
class ReplaceOperation(Operation):
    symbol = attr.ib()

@attr.s
class PopOperation(Operation):
    pass
