import math

import torch

class Layer(torch.nn.Module):

    def __init__(self, input_size, output_size, activation=None):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.activation = activation

    def forward(self, x):
        result = self.linear(x)
        if self.activation is not None:
            result = self.activation(result)
        return result

    def get_nonlinearity(self):
        a = self.activation
        if a is None or isinstance(a, torch.nn.Softmax):
            return 'linear'
        else:
            return type(a).__name__.lower()

    def xavier_uniform_init(self, generator=None):
        gain = torch.nn.init.calculate_gain(self.get_nonlinearity())
        xavier_uniform_(self.linear.weight, gain, generator=generator)
        torch.nn.init.constant_(self.linear.bias, 0.0)

    def input_size(self):
        return self.linear.in_features

    def output_size(self):
        return self.linear.out_features

class FeedForward(torch.nn.Sequential):

    def __init__(self, input_size, layer_sizes, activation):
        modules = []
        for layer_size in layer_sizes:
            modules.append(Layer(input_size, layer_size, activation))
            input_size = layer_size
        super().__init__(*modules)

    def input_size(self):
        return self[0].input_size()

    def output_size(self):
        return self[-1].output_size()

class MultiLayer(Layer):

    def __init__(self, input_size, output_size, n, activation=None):
        super().__init__(input_size, output_size * n, activation)
        self.n = n

    def forward(self, x):
        result = self.linear(x)
        size = result.size()[:-1] + (self.n, -1)
        result = result.view(size)
        if self.activation is not None:
            result = self.activation(result)
        return result

def xavier_uniform_(tensor, gain, generator):
    # A rewrite of torch.nn.init.xavier_uniform_() that supports a manual seed
    # as well as n-dimensional tensors.
    fan_out = tensor.size(0)
    fan_in = tensor.size(-1)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-a, a, generator=generator)
