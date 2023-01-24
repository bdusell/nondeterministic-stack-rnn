import functools
import typing

import torch

from torch_rnn_tools import UnidirectionalRNN

class StackRNNBase(UnidirectionalRNN):

    def __init__(self, input_size, stack_reading_size, controller):
        super().__init__()
        self._input_size = input_size
        self.stack_reading_size = stack_reading_size
        self.controller = controller(input_size + stack_reading_size)

    def input_size(self):
        return self._input_size

    def output_size(self):
        return self.controller.output_size()

    class State(UnidirectionalRNN.State):

        def __init__(self, rnn, hidden_state, previous_stack, return_actions,
                previous_actions, return_readings, previous_reading,
                stack_args, stack_kwargs):
            super().__init__()
            self.rnn = rnn
            self.hidden_state = hidden_state
            self.previous_stack = previous_stack
            self.return_actions = return_actions
            self.previous_actions = previous_actions
            self.return_readings = return_readings
            self.previous_reading = previous_reading
            self.stack_args = stack_args
            self.stack_kwargs = stack_kwargs

        def next(self, input_tensor):
            if self.previous_stack is None:
                stack = self.rnn.initial_stack(
                    self.hidden_state.batch_size(),
                    self.rnn.stack_reading_size,
                    *self.stack_args,
                    **self.stack_kwargs
                )
                actions = None
            else:
                stack, actions = self.compute_stack(
                    self.hidden_state_output,
                    self.previous_stack
                )
            reading = stack.reading()
            controller_input = torch.cat((input_tensor, reading), dim=1)
            next_hidden_state = self.hidden_state.next(controller_input)
            return self.rnn.State(
                rnn=self.rnn,
                hidden_state=next_hidden_state,
                previous_stack=stack,
                return_actions=self.return_actions,
                previous_actions=actions if self.return_actions else None,
                return_readings=self.return_readings,
                previous_reading=reading if self.return_readings else None,
                stack_args=None,
                stack_kwargs=None
            )

        def output(self):
            output = self.hidden_state_output
            extras = []
            if self.return_actions:
                extras.append(self.previous_actions)
            if self.return_readings:
                extras.append(self.previous_reading)
            if extras:
                return (output, *extras)
            else:
                return output

        @functools.cached_property
        def hidden_state_output(self):
            return self.hidden_state.output()

        def detach(self):
            return self.rnn.State(
                rnn=self.rnn,
                hidden_state=self.hidden_state.detach(),
                previous_stack=self.previous_stack.detach() if self.previous_stack is not None else None,
                return_actions=self.return_actions,
                # Do not detach previous_actions because its type is not always
                # Tensor (e.g. it might be a tuple of tensors). This is okay
                # because it will not be used for future hidden states anyway;
                # it will only be returned in the output of the next state, so
                # it doesn't really matter if it's not detached.
                previous_actions=self.previous_actions,
                return_readings=self.return_readings,
                previous_reading=self.previous_reading,
                stack_args=self.stack_args,
                stack_kwargs=self.stack_kwargs
            )

        def batch_size(self):
            return self.hidden_state.batch_size()

        def slice_batch(self, s):
            return self.rnn.State(
                rnn=self.rnn,
                hidden_state=self.hidden_state.slice_batch(s),
                previous_stack=self.previous_stack.slice_batch(s) if self.previous_stack is not None else None,
                return_actions=self.return_actions,
                previous_actions=self.previous_actions,
                return_readings=self.return_readings,
                previous_reading=self.previous_reading,
                stack_args=self.stack_args,
                stack_kwargs=self.stack_kwargs
            )

        def compute_stack(self, hidden_state, stack):
            raise NotImplementedError

    def initial_state(self,
            batch_size: int,
            *args,
            return_actions: bool=False,
            return_readings: bool=False,
            first_layer: typing.Optional[torch.Tensor]=None,
            **kwargs):
        """Get the initial state of the stack RNN.

        :param return_actions: If true, then the output at each timestep will
            also include the stack actions that were emitted just before the
            current timestep. Note that the actions for timesteps 0 and 1 are
            always ``None``.
        :param return_readings: If true, then the output at each timestep will
            also include the stack reading that was emitted just before the
            current timestep. Note that the stack reading for timestep 0 is
            always ``None``.
        :param first_layer: Will be passed to the controller.
        :param args: Will be passed to :py:meth:`initial_stack`.
        :param kwargs: Will be passed to :py:meth:`initial_stack`.
        """
        return self.State(
            rnn=self,
            hidden_state=self.controller.initial_state(
                batch_size,
                first_layer=first_layer
            ),
            # There is no "previous stack" for the initial hidden state, so
            # set it None. It will call initial_stack() to supply the stack for
            # the next timestep.
            previous_stack=None,
            return_actions=return_actions,
            previous_actions=None,
            return_readings=return_readings,
            previous_reading=None,
            stack_args=args,
            stack_kwargs=kwargs
        )

    def initial_stack(self, batch_size, reading_size, *args, **kwargs):
        raise NotImplementedError
