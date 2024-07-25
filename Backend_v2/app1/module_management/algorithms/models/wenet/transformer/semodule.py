import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple



class Swish(nn.Module):
    r"""
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class SEModule(nn.Module):
    r"""
    Squeeze-and-excitation module.

    Args:
        dim (int): Dimension to be used for two fully connected (FC) layers

    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``

    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, dim: int) -> None:
        super(SEModule, self).__init__()
        assert dim % 8 == 0, 'Dimension should be divisible by 8.'

        self.dim = dim
        self.sequential = nn.Sequential(
            nn.Linear(dim, dim // 8),
            Swish(),
            nn.Linear(dim // 8, dim),
        )

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for SE Layer.

        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(seq_length)``

        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        inputs = inputs.permute(0, 2, 1)
        residual = inputs
        seq_lengths = inputs.size(2)

        inputs = inputs.sum(dim=2) / input_lengths.unsqueeze(1)
        output = self.sequential(inputs)

        output = output.sigmoid().unsqueeze(2)
        output = output.repeat(1, 1, seq_lengths)
        output = output*residual
        output = output.permute(0, 2, 1)
        return output
