import torch

from ..pytorch_tools.unidirectional_rnn import UnidirectionalLSTM
from ..pytorch_tools.layer import Layer, MultiLayer
from .common import StackRNNBase

class GrefenstetteRNN(StackRNNBase):

    def __init__(self, input_size, hidden_units, stack_embedding_size,
            synchronized, controller=UnidirectionalLSTM):
        super().__init__(
            input_size, hidden_units, stack_embedding_size, controller)
        if synchronized:
            action_layers = 1
        else:
            action_layers = stack_embedding_size
        self.action_layer = MultiLayer(hidden_units, 2, action_layers,
            torch.nn.Sigmoid())
        self.push_value_layer = Layer(hidden_units, stack_embedding_size,
            torch.nn.Tanh())

    def initial_stack(self, batch_size, stack_size, device):
        return GrefenstetteStack(
            elements=[],
            bottom=torch.zeros(batch_size, self.stack_size, device=device)
        )

    class State(StackRNNBase.State):

        def compute_stack(self, hidden_state, stack):
            actions = self.rnn.action_layer(hidden_state)
            push_value = self.rnn.push_value_layer(hidden_state)
            return stack.next(actions, push_value)

class GrefenstetteStack:

    def __init__(self, elements, bottom):
        self.elements = elements
        self.bottom = bottom

    def reading(self):
        device = self.bottom.device
        batch_size = self.bottom.size(0)
        result = self.bottom
        strength_left = torch.ones(batch_size, 1, device=device)
        for value, strength in reversed(self.elements):
            result = result + value * torch.min(
                strength,
                torch.nn.functional.relu(strength_left)
            )
            strength_left = strength_left - strength
        return result

    def next(self, actions, push_value):
        return GrefenstetteStack(
            self.next_elements(actions, push_value),
            self.bottom
        )

    def next_elements(self, actions, push_value):
        push_strength = actions[:, :, :1].squeeze(-1)
        pop_strength = actions[:, :, 1:].squeeze(-1)
        result = []
        strength_left = pop_strength
        for value, strength in reversed(self.elements):
            result.append((
                value,
                torch.nn.functional.relu(
                    strength -
                    torch.nn.functional.relu(strength_left)
                )
            ))
            strength_left = strength_left - strength
        result.reverse()
        result.append((push_value, push_strength))
        return result
