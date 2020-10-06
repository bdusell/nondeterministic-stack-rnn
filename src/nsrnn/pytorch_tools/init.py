import torch

from .layer import Layer

def default_fallback(name, data, generator):
    data.uniform_(generator=generator)

def _raise_return_error():
    raise TypeError(
        'initialize() should return an iterable of '
        '`torch.nn.Parameter`s containing the parameters that were '
        'initialized and that should be skipped by the fallback '
        'initializer (almost always this is just `module.parameters()`)')

def init_modules(module, initialize, fallback=default_fallback, generator=None):
    seen = set()
    def recurse(module):
        for submodule_name, submodule in module.named_children():
            matched, params = initialize(submodule_name, submodule, generator)
            if matched:
                if params is None:
                    params = submodule.parameters()
                for param in params:
                    if not isinstance(param, torch.nn.Parameter):
                        _raise_return_error()
                    seen.add(id(param))
            else:
                recurse(submodule)
    recurse(module)
    for param_name, param in module.named_parameters():
        if id(param) not in seen:
            fallback(param_name, param.data, generator)

def init_modules_by_type(module, functions, fallback=default_fallback,
        generator=None):
    if isinstance(functions, dict):
        def initialize(name, module, generator):
            func = functions.get(type(module))
            if func is not None:
                return True, func(name, module, generator)
            else:
                return False, None
    else:
        if not isinstance(functions, (list, tuple)):
            functions = tuple(functions)
        def initialize(name, module, generator):
            for type_, func in functions:
                if isinstance(module, type_):
                    return True, func(name, module, generator)
            return False, None
    init_modules(module, initialize, fallback, generator)

def xavier_uniform_init(module, generator, fallback=default_fallback):
    def init_layer(name, module, generator):
        module.xavier_uniform_init(generator=generator)
    init_modules_by_type(module, [(Layer, init_layer)], fallback)
