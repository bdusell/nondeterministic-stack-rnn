import typing

import torch

from .unidirectional_rnn import UnidirectionalRNN
from .wrapper import Wrapper, handle_packed_sequence
from .dropout import InputDropoutWrapper, OutputDropoutWrapper
from .output import OutputLayerWrapper

class EmbeddingWrapper(Wrapper):
    """Adds a word embedding layer to an RNN."""

    def __init__(self,
            rnn: UnidirectionalRNN,
            vocabulary_size: int,
            padding_index: typing.Optional[int]=None
        ):
        """
        :param rnn: The wrapped RNN.
        :vocabulary_size: The number of word types in the embedding layer.
        :padding_index: Optional padding index.
        """
        super().__init__(rnn)
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=rnn.input_size(),
            padding_idx=padding_index
        )

    def transform_input(self, x):
        return handle_packed_sequence(self.embedding_layer, x)

    def input_size(self):
        raise TypeError('EmbeddingWrapper does not have an input_size')

class TiedEmbeddingWrapper(Wrapper):
    """Adds tied input and output word embedding layers to an RNN."""

    def __init__(self,
            rnn: UnidirectionalRNN,
            vocabulary_size: int,
            input_slice: slice=slice(None),
            output_slice: slice=slice(None),
            bias: bool=False
        ):
        """
        :param rnn: The wrapped RNN.
        :param vocabulary_size: The size of the vocabulary (number of
            embedding vectors learned).
        :param input_slice: A slice of the embedding matrix to be used as the
            input embeddings. Allows the input and output vocabularies to be
            different sizes.
        :param output_slice: A slice of the embedding matrix to be used as the
            output embeddings. Allows the input and output vocabularies to be
            different sizes.
        :param bias: Whether to include a bias term in the output layer.
            The Inan et al. 2017 paper does not include a bias term; the code
            for the AWD-LSTM does.
        """
        # The embedding size needs to match the input and output size of the
        # wrapped RNN.
        rnn_input_size = rnn.input_size()
        rnn_output_size = rnn.output_size()
        if rnn_input_size != rnn_output_size:
            raise ValueError(
                f'the input size ({rnn_input_size}) and output size '
                f'({rnn_output_size}) of an RNN wrapped in tied word '
                f'embeddings must be equal')
        embedding_size = rnn_input_size
        super().__init__(rnn)
        self.input_slice = input_slice
        self.output_slice = output_slice
        # self.embeddings : vocabulary_size x embedding_size
        self.embeddings = torch.nn.Parameter(
            torch.zeros(
                (vocabulary_size, embedding_size),
                device=self.device))
        self._output_size = self.embeddings[output_slice, :].size(0)
        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(self._output_size, device=self.device))
        else:
            self.bias = None

    def transform_input(self, x):
        # x : batch_size x sequence_length in [0, V)
        # weight is expected to be vocabulary_size x embedding_size
        # return : batch_size x sequence_length x embedding_size
        return torch.nn.functional.embedding(
            input=x,
            weight=self.embeddings[self.input_slice, :]
        )

    def transform_output(self, y):
        # y : batch_size x sequence_length x embedding_size
        # weight is expected to be vocabulary_size x embedding_size
        # return : batch_size x sequence_length x vocabulary_size
        return torch.nn.functional.linear(
            input=y,
            weight=self.embeddings[self.output_slice, :],
            bias=self.bias
        )

    def input_size(self):
        raise TypeError('TiedEmbeddingWrapper does not have an input_size')

    def output_size(self):
        return self._output_size

def with_embeddings(
        rnn: UnidirectionalRNN,
        vocabulary_size: int,
        tied: bool=True,
        output_size: typing.Optional[int]=None,
        input_slice: slice=slice(None),
        output_slice: slice=slice(None),
        bias: bool=False,
        input_dropout: typing.Optional[float]=None,
        output_dropout: typing.Optional[float]=None
    ) -> UnidirectionalRNN:
    """Wrap an RNN with a word embedding layer and an output layer.

    :param rnn: The wrapped RNN.
    :param vocabulary_size: The size of the vocabulary (number of embedding
        vectors learned).
    :param tied: Whether to tie the input and output weights.
    :param output_size: The desired size of the output layer. This is useful
        for setting the size of the output when ``tied=False``. When
        ``tied=True``, this is determined by ``output_slice`` bt default. When
        ``tied=False``, this is ``vocabulary_size`` by default.
    :param input_slice: See :py:class:`TiedEmbeddingWrapper.__init__`.
    :param output_slice: See :py:class:`TiedEmbeddingWrapper.__init__`.
    :param bias: Whether to include a bias term in the output layer.
    :param input_dropout: Optional dropout applied between the input word
        embeddings and the input to the RNN.
    :param output_dropout: Optional dropout applied between the output of the
        RNN and the output layer.
    :return: A new RNN that wraps the original RNN with embedding and output
        layers.
    """
    rnn = InputDropoutWrapper(rnn, input_dropout)
    rnn = OutputDropoutWrapper(rnn, output_dropout)
    if tied:
        rnn = TiedEmbeddingWrapper(
            rnn,
            vocabulary_size,
            input_slice,
            output_slice,
            bias,
        )
        if output_size is not None:
            rnn_output_size = rnn.output_size()
            if rnn_output_size != output_size:
                raise ValueError(
                    f'the output slice ({output_slice}) for the tied word '
                    f'embedding layer does not produce the desired output '
                    f'size ({output_size}) for the tied word embeddings')
    else:
        if output_size is None:
            output_size = vocabulary_size
        rnn = EmbeddingWrapper(rnn, vocabulary_size)
        rnn = OutputLayerWrapper(rnn, output_size=output_size, bias=bias)
    return rnn
