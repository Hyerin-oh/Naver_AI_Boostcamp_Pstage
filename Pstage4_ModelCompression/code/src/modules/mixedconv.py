import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils.torch_utils import make_divisible
from src.modules.base_generator import GeneratorAbstract

import math

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'Swish': Swish(),
}

def _RoundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c

def _SplitChannels(channels, num_groups):
    split_channels = [channels//num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels

def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        NON_LINEARITY[non_linear]
    )

def Conv1x1Bn(in_channels, out_channels, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        NON_LINEARITY[non_linear]
    )

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, squeeze_factor=4):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = make_divisible(squeeze_channels // squeeze_factor, 8)
        # if not squeeze_channels.is_integer():
        #     raise ValueError('channels must be divisible by 1/ratio')

        # squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = NON_LINEARITY['Swish']
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y

class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GroupedConv2d, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_in_channels = _SplitChannels(in_channels, self.num_groups)
        self.split_out_channels = _SplitChannels(out_channels, self.num_groups)

        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(nn.Conv2d(
                self.split_in_channels[i],
                self.split_out_channels[i],
                kernel_size[i],
                stride=stride,
                padding=padding,
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)

        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x

class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_channels = _SplitChannels(channels, self.num_groups)

        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i]//2,
                groups=self.split_channels[i],
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x

class MixNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[3],
        expand_ksize=[1],
        project_ksize=[1],
        stride=1,
        expand_ratio=1,
        non_linear='ReLU',
        use_se=1
    ):

        super(MixNetBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        
        self.residual_connection = (stride == 1 and in_channels == out_channels)

        conv = []

        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                GroupedConv2d(in_channels, expand_channels, expand_ksize),
                nn.BatchNorm2d(expand_channels),
                NON_LINEARITY[non_linear]
            )
            conv.append(pw_expansion)

        # depthwise convolution phase
        dw = nn.Sequential(
            MDConv(expand_channels, kernel_size, stride),
            nn.BatchNorm2d(expand_channels),
            NON_LINEARITY[non_linear]
        )
        conv.append(dw)

        if use_se:
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels)
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            GroupedConv2d(expand_channels, out_channels, project_ksize),
            nn.BatchNorm2d(out_channels)
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MixNetBlockGenerator(GeneratorAbstract):
    """MixNet block generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self._get_divisible_channel(self.args[2] * self.width_multiply)

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from src.common_modules based on the class name."""
        return getattr(__import__("src.modules", fromlist=[""]), self.name)

    def __call__(self, repeat: int = 1):
        """call method.

        InvertedResidualv3 args consists,
        repeat(=n), [kernel, exp_ratio, out, SE, NL, s] //
        note original notation from paper is [exp_size, out, SE, NL, s]
        """
        module = []
        kernels, t, _, se, s, non_linear = self.args  # c is equivalent as self.out_channel
        inp, oup = self.in_channel, self.out_channel
        for i in range(repeat):
            stride = s if i == 0 else 1
            module.append(
                self.base_module(
                    in_channels=inp,
                    out_channels=oup,
                    kernel_size=kernels,
                    stride=stride,
                    use_se=se,
                    expand_ratio=t,
                    non_linear=non_linear
                )
            )
            inp = oup
        return self._get_module(module)