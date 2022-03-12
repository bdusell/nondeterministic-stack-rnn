from __future__ import annotations
import typing

import torch

from ._utils import apply_to_first_element

class UnidirectionalRNN(torch.nn.Module):
    """An API for unidirectional RNNs."""

    def forward(self,
            input_sequence: torch.Tensor,
            *args,
            initial_state: typing.Optional['UnidirectionalRNN.State']=None,
            return_state: bool=False,
            include_first: bool=True,
            **kwargs
        ) -> typing.Union[torch.Tensor, tuple]:
        r"""Let :math:`B` be batch size and :math:`n` be the length of the
        input sequence.

        :param input_sequence: A :math:`B \times n \times \cdots` tensor that
            serves as the input to the RNN.
        :param initial_state: An optional initial state to use instead of the
            default initial state created by :py:meth:`initial_state`.
        :param return_state: Whether to return the last :py:class:`State` of
            the RNN as an additional output. This state might be used to
            initialize a subsequent run.
        :param include_first: Whether to include an output corresponding to a
            prediction for the first element in the input. If ``include_first``
            is true, then the length of the output tensor will be
            :math:`n + 1`. Otherwise, it will be :math:`n`.
        :param args: Extra arguments passed to :py:meth:`initial_state`.
        :param kwargs: Extra arguments passed to :py:meth:`initial_state`.
        :return: A :py:class:`~torch.Tensor` or a :py:class:`tuple` whose
            first element is a tensor. The tensor is the output of the RNN.
            Additional outputs may be returned in extra elements in the tuple.
        """
        # input_sequence: B x n x ...
        if initial_state is not None:
            if not isinstance(initial_state, self.State):
                raise TypeError(f'initial_state must be of type {self.State.__name__}')
            state = initial_state
        else:
            batch_size = input_sequence.size(0)
            state = self.initial_state(batch_size, *args, **kwargs)
        # return : B x n x ...
        return state.forward(
            input_sequence,
            return_state=return_state,
            include_first=include_first)

    def input_size(self) -> int:
        """Get the size of the vector inputs to the RNN."""
        raise NotImplementedError

    def output_size(self) -> int:
        """Get the size of the vector outputs from the RNN."""
        raise NotImplementedError

    def wrapped_rnn(self):
        """If this RNN serves as a wrapper around another, return the wrapped
        RNN. By default this returns ``self``."""
        return self

    def wrap_input(self, x):
        """If this RNN serves as a wrapper around another, take an input and
        return the transformed input that would be passed to the wrapped
        RNN. Otherwise return the input unchanged."""
        return x

    class State:
        """Represents the hidden state of the RNN at some point during
        processing of the input sequence."""

        def next(self, input_tensor: torch.Tensor) -> State:
            r"""Feed an input to this hidden state and produce the next hidden
            state.

            :param input_tensor: A :math:`B \times d` tensor, where :math:`B`
                is batch size and :math:`d` is the RNN's ``input_size``.
            """
            raise NotImplementedError

        def output(self) -> typing.Union[torch.Tensor, tuple]:
            r"""Get the output tensor associated with this state.

            For example, this can be the hidden state vector itself, or the
            hidden state passed through an affine transformation.

            :return: A :math:`B \times \cdots` tensor, or a tuple whose first
                element is a tensor. Remaining elements will be transposed
                into lists and returned from :py:meth:`outputs` or
                :py:meth:`UnidirectionalRNN.forward`.
            """
            raise NotImplementedError

        def detach(self) -> State:
            """Return a copy of this state with all tensors detached."""
            return self.transform_tensors(lambda x: x.detach())

        def batch_size(self) -> int:
            """Get the batch size of the tensors in this state."""
            raise NotImplementedError

        def slice_batch(self, s: slice) -> State:
            """Return a copy of this state with only certain batch elements
            included, determined by the slice ``s``.
            
            :param s: The slice object used to determine which batch elements
                to keep.
            """
            return self.transform_tensors(lambda x: x[s, ...])

        def transform_tensors(self,
                func: typing.Callable[[torch.Tensor], torch.Tensor]) -> State:
            """Return a copy of this state with all tensors passed through a
            function."""
            raise NotImplementedError

        def fastforward(self, input_sequence: torch.Tensor) -> State:
            r"""Feed a sequence of inputs to the state and return the resulting
            state.

            :param input_sequence: A :math:`B \times n \times \cots` tensor,
                representing :math:`n` input tensors.
            :return: New state.
            """
            state = self
            for input_tensor in input_sequence.transpose(0, 1):
                state = state.next(input_tensor)
            return state

        def states(self,
                input_sequence: torch.Tensor,
                include_first: bool
            ) -> typing.Iterable[State]:
            r"""Feed a sequence of inputs to the state and generate all the
            states produced after each input.

            :param input_sequence: A :math:`B \times n \times \cdots` tensor.
            :param include_first: Whether to include ``self`` as the first
                state in the returned sequence of states.
            :return: Sequence of states.
            """
            state = self
            if include_first:
                yield state
            for input_tensor in input_sequence.transpose(0, 1):
                state = state.next(input_tensor)
                yield state

        def outputs(self,
                input_sequence: torch.Tensor,
                include_first: bool
            ) -> typing.Iterable[typing.Union[torch.Tensor, tuple]]:
            """Like :py:meth:`states`, but return the states' outputs."""
            for state in self.states(input_sequence, include_first):
                yield state.output()

        def forward(self,
                input_sequence: torch.Tensor,
                return_state: bool,
                include_first: bool
            ) -> typing.Union[torch.Tensor, tuple]:
            """Like :py:meth:`UnidirectionalRNN.forward`, but start with this
            state as the initial state."""
            if return_state:
                outputs = []
                for state in self.states(input_sequence, include_first):
                    outputs.append(state.output())
                return stack_outputs(outputs), state
            else:
                return stack_outputs(self.outputs(input_sequence, include_first))

    def initial_state(self, batch_size: int, *args, **kwargs) -> State:
        """Get the initial state of the RNN.

        :param batch_size: Batch size.
        :return: A state.
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Get this module's device."""
        return next(self.parameters()).device

def stack_outputs(outputs):
    it = iter(outputs)
    first = next(it)
    if isinstance(first, tuple):
        output, *extra = first
        output_list = [output]
        extra_lists = tuple([e] for e in extra)
        for output, *extra in it:
            output_list.append(output)
            for extra_list, extra_item in zip(extra_lists, extra):
                extra_list.append(extra_item)
        return (torch.stack(output_list, dim=1), *extra_lists)
    else:
        output_list = [first]
        output_list.extend(it)
        return torch.stack(output_list, dim=1)
