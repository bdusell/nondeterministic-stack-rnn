import torch

from ..pytorch_tools.unidirectional_rnn import UnidirectionalLSTM
from ..pytorch_tools.layer import Layer, MultiLayer
from .common import StackRNNBase

def construct_stack(batch_size, stack_size, sequence_length, device):
    return JoulinMikolovStack(
        elements=torch.zeros(batch_size, stack_size, 0, device=device),
        timestep=1,
        sequence_length=sequence_length
    )

PUSH = 0
POP = 1
NOOP = 2

class JoulinMikolovRNN(StackRNNBase):

    PUSH = PUSH
    POP = POP
    NOOP = NOOP

    num_actions = 3

    def __init__(self, input_size, hidden_units, stack_embedding_size,
            synchronized, controller=UnidirectionalLSTM):
        super().__init__(
            input_size, hidden_units, stack_embedding_size, controller)
        if synchronized:
            action_layers = 1
        else:
            action_layers = stack_embedding_size
        self.action_layer = MultiLayer(hidden_units, self.num_actions, action_layers,
            torch.nn.Softmax(dim=2))
        self.push_value_layer = Layer(hidden_units, stack_embedding_size,
            torch.nn.Sigmoid())

    def generate_outputs(self, input_tensors, *args, **kwargs):
        # Automatically use the sequence length to optimize the stack
        # computation.
        return super().generate_outputs(
            input_tensors,
            sequence_length=input_tensors.size(0),
            *args,
            **kwargs)

    def initial_stack(self, batch_size, stack_size, device,
            sequence_length=None, stack_constructor=construct_stack):
        """
        If the sequence length is known, passing it via `sequence_length` can
        be used to reduce the time and space required by the stack by half.
        """
        return stack_constructor(batch_size, stack_size, sequence_length, device)

    class State(StackRNNBase.State):

        def compute_stack(self, hidden_state, stack):
            actions = self.rnn.action_layer(hidden_state)
            # num_stacks = 1 or stack_size
            # actions : batch_size x num_stacks x 2
            # push_value : batch_size x stack_size
            self.signals = actions
            push_prob = actions[:, :, PUSH:PUSH+1]
            pop_prob = actions[:, :, POP:POP+1]
            noop_prob = actions[:, :, NOOP:NOOP+1]
            push_value = self.rnn.push_value_layer(hidden_state)
            return stack.next(push_prob, pop_prob, noop_prob, push_value)

class JoulinMikolovStack:

    def __init__(self, elements, timestep=1, sequence_length=None):
        self.elements = elements
        self.sequence_length = sequence_length
        self.timestep = timestep

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
            self.sequence_length
        )

    def next_elements(self, push_prob, pop_prob, noop_prob, push_value):
        # push_prob : batch_size x num_stacks x 1
        # pop_prob : batch_size x num_stacks x 1
        batch_size, stack_size, _ = self.elements.size()
        device = self.elements.device
        actual_stack_height = self.timestep
        if self.sequence_length is not None:
            actual_stack_height = min(
                actual_stack_height,
                self.sequence_length - self.timestep)
        max_push_elements = actual_stack_height - 1
        # self.elements : batch_size x stack_size x stack_height
        push_elements = self.elements
        if push_elements.size(2) > max_push_elements:
            push_elements = push_elements[:, :, :max_push_elements]
        push_terms = push_prob * torch.cat((
            push_value.unsqueeze(2),
            push_elements
        ), dim=2)
        # push_terms : batch_size x stack_size x stack_height
        pop_terms = pop_prob * self.elements[:, :, 1:1+actual_stack_height]
        # pop_terms : batch_size x stack_size x stack_height
        noop_terms = noop_prob * self.elements[:, :, :actual_stack_height]
        # noop_terms : batch_size x stack_size x stack_height
        return jagged_sum(jagged_sum(push_terms, noop_terms), pop_terms)

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
