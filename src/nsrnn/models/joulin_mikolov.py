import math
import typing

import attr
import torch

from torch_rnn_tools import UnidirectionalLSTM
from torch_extras.layer import Layer, MultiLayer
from .common import StackRNNBase

PUSH = 0
POP = 1
NOOP = 2

def construct_stack(batch_size, stack_size, max_sequence_length, max_depth, device):
    return JoulinMikolovStack(
        elements=torch.zeros(batch_size, stack_size, 0, device=device),
        timestep=0,
        max_sequence_length=max_sequence_length,
        max_depth=max_depth
    )

class JoulinMikolovRNN(StackRNNBase):

    PUSH = PUSH
    POP = POP
    NOOP = NOOP

    NUM_ACTIONS = 3

    def __init__(self,
            input_size: int,
            stack_embedding_size: int,
            controller: typing.Callable,
            push_hidden_state: bool=False,
            stack_depth_limit: typing.Union[int, typing.Literal[math.inf]]=math.inf
        ):
        """
        :param input_size: Size the input vectors to the RNN.
        :param stack_embedding_size: Number of elements in each vector element
            in the stack.
        :param controller: Constructor for the RNN controller.
        :param push_hidden_state: Whether to push the hidden state of the
            controller or to learn a projection for pushed vectors
            automatically.
        """
        super().__init__(input_size, stack_embedding_size, controller)
        self.action_layer = Layer(
            self.controller.output_size(),
            self.NUM_ACTIONS,
            torch.nn.Softmax(dim=1)
        )
        if push_hidden_state:
            hidden_state_size = self.controller.output_size()
            if stack_embedding_size != hidden_state_size:
                raise ValueError(
                    f'push_hidden_state is True, but stack_embedding_size '
                    f'({stack_embedding_size}) does not match the output size '
                    f'of the controller ({hidden_state_size})')
            self.push_value_layer = torch.nn.Identity()
        else:
            self.push_value_layer = Layer(
                self.controller.output_size(),
                stack_embedding_size,
                torch.nn.Sigmoid()
            )
        self.stack_depth_limit = stack_depth_limit

    def forward(self, input_sequence, *args, return_state=False, **kwargs):
        # Automatically use the sequence length to optimize the stack
        # computation. Don't use it if returning the stack state.
        max_sequence_length = math.inf if return_state else input_sequence.size(1)
        return super().forward(
            input_sequence,
            *args,
            return_state=return_state,
            max_sequence_length=max_sequence_length,
            **kwargs)

    class State(StackRNNBase.State):

        def compute_stack(self, hidden_state, stack):
            actions = self.rnn.action_layer(hidden_state)
            # actions : batch_size x NUM_ACTIONS
            # push_value : batch_size x stack_embedding_size
            self.signals = actions
            push_prob = actions[:, PUSH]
            pop_prob = actions[:, POP]
            noop_prob = actions[:, NOOP]
            push_value = self.rnn.push_value_layer(hidden_state)
            return stack.next(push_prob, pop_prob, noop_prob, push_value)

    def initial_stack(self,
            batch_size,
            stack_size,
            max_sequence_length: typing.Union[int, typing.Literal[math.inf]]=math.inf,
            stack_constructor=construct_stack
        ):
        """
        If the sequence length is known, passing it via `max_sequence_length`
        can be used to reduce the time and space required by the stack by half.
        """
        return stack_constructor(
            batch_size,
            stack_size,
            max_sequence_length,
            self.stack_depth_limit,
            self.device)

class JoulinMikolovStack:

    def __init__(self, elements, timestep, max_sequence_length, max_depth):
        # elements : batch_size x stack_embedding_size x stack_height
        self.elements = elements
        self.timestep = timestep
        self.max_sequence_length = max_sequence_length
        self.max_depth = max_depth

    def reading(self):
        batch_size, stack_size, num_elements = self.elements.size()
        if num_elements > 0:
            return self.elements[:, :, 0]
        else:
            return torch.zeros(batch_size, stack_size, device=self.elements.device)

    def next(self, push_prob, pop_prob, noop_prob, push_value):
        return JoulinMikolovStack(
            self.next_elements(push_prob, pop_prob, noop_prob, push_value),
            self.timestep + 1,
            self.max_sequence_length,
            self.max_depth
        )

    def next_elements(self, push_prob, pop_prob, noop_prob, push_value):
        # push_prob : batch_size
        # pop_prob : batch_size
        # noop_prob : batch_size
        # push_value : batch_size x stack_embedding_size
        # self.elements : batch_size x stack_embedding_size x stack_height
        batch_size = self.elements.size(0)
        device = self.elements.device
        next_timestep = self.timestep + 1
        actual_stack_height = min(
            next_timestep,
            self.max_sequence_length - next_timestep,
            self.max_depth
        )
        max_push_elements = actual_stack_height - 1
        push_elements = self.elements
        if push_elements.size(2) > max_push_elements:
            push_elements = push_elements[:, :, :max_push_elements]
        push_terms = push_prob[:, None, None] * torch.cat([
            push_value[:, :, None],
            push_elements
        ], dim=2)
        # push_terms : batch_size x stack_embedding_size x stack_height
        pop_terms = pop_prob[:, None, None] * self.elements[:, :, 1:1+actual_stack_height]
        # pop_terms : batch_size x stack_embedding_size x stack_height
        noop_terms = noop_prob[:, None, None] * self.elements[:, :, :actual_stack_height]
        # noop_terms : batch_size x stack_embedding_size x stack_height
        return jagged_sum(jagged_sum(push_terms, noop_terms), pop_terms)

    def detach(self):
        return self.transform_tensors(lambda x: x.detach())

    def slice_batch(self, s):
        return self.transform_tensors(lambda x: x[s, ...])

    def transform_tensors(self, func):
        return JoulinMikolovStack(
            func(self.elements),
            self.timestep,
            self.max_sequence_length,
            self.max_depth
        )

def jagged_sum(a, b):
    # Efficiently adds two stack tensors which may not have the same number
    # of stack elements.
    # Precondition: a.size(2) >= b.size(2)
    a_size = a.size(2)
    b_size = b.size(2)
    if b_size == 0:
        # This branch is needed because .backward() throws an exception
        # for some reason when b_size is 0.
        return a
    elif a_size == b_size:
        return a + b
    else:
        return torch.cat([
            a[:, :, :b_size] + b,
            a[:, :, b_size:]
        ], dim=2)
