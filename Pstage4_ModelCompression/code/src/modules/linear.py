from typing import Union

import torch
from torch import nn as nn

from src.modules.base_generator import GeneratorAbstract
from src.utils.torch_utils import Activation


class Linear(nn.Module):
    """Linear module."""

    def __init__(self, in_channel: int, out_channel: int, activation: Union[str, None], dropout_p :float):
        """

        Args:
            in_channel: input channels.
            out_channel: output channels.
            activation: activation name. If None is given, nn.Identity is applied
                which is no activation.
        """
        super().__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.activation = Activation(activation)()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.activation(self.linear(self.dropout(x)))


class LinearGenerator(GeneratorAbstract):
    """Linear (fully connected) module generator for parsing."""

    def __init__(self, *args, **kwargs):
        """Initailize."""
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self.args[0]

    def __call__(self, repeat: int = 1):

        act = self.args[1] if len(self.args) > 1 else None
        # dropout probability 가 입력 되지 않을 경우 0로 고정
        dropout_p = self.args[2] if len(self.args) > 2 else 0.0 

        return self._get_module(
            Linear(self.in_channel, self.out_channel, activation=act, dropout_p = dropout_p)
        )
