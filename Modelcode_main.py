from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# backbone

__all__ = ['MobileNetV4ConvLarge', 'MobileNetV4ConvSmall', 'MobileNetV4ConvMedium', 'MobileNetV4HybridMedium',
           'MobileNetV4HybridLarge']

MNV4ConvSmall_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [32, 32, 3, 2],
            [32, 32, 1, 1]
        ]
    },
    "layer2": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [32, 96, 3, 2],
            [96, 64, 1, 1]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            [64, 96, 5, 5, True, 2, 3],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 3, 0, True, 1, 4],
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            [96, 128, 3, 3, True, 2, 6],
            [128, 128, 5, 5, True, 1, 4],
            [128, 128, 0, 5, True, 1, 4],
            [128, 128, 0, 5, True, 1, 3],
            [128, 128, 0, 3, True, 1, 4],
            [128, 128, 0, 3, True, 1, 4],
        ]
    },
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [128, 960, 1, 1],
            [960, 1280, 1, 1]
        ]
    }
}

MNV4ConvMedium_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 1]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [32, 48, 2, 4.0, True]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 80, 3, 5, True, 2, 4],
            [80, 80, 3, 3, True, 1, 2]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 8,
        "block_specs": [
            [80, 160, 3, 5, True, 2, 6],
            [160, 160, 3, 3, True, 1, 4],
            [160, 160, 3, 3, True, 1, 4],
            [160, 160, 3, 5, True, 1, 4],
            [160, 160, 3, 3, True, 1, 4],
            [160, 160, 3, 0, True, 1, 4],
            [160, 160, 0, 0, True, 1, 2],
            [160, 160, 3, 0, True, 1, 4]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            [160, 256, 5, 5, True, 2, 6],
            [256, 256, 5, 5, True, 1, 4],
            [256, 256, 3, 5, True, 1, 4],
            [256, 256, 3, 5, True, 1, 4],
            [256, 256, 0, 0, True, 1, 4],
            [256, 256, 3, 0, True, 1, 4],
            [256, 256, 3, 5, True, 1, 2],
            [256, 256, 5, 5, True, 1, 4],
            [256, 256, 0, 0, True, 1, 4],
            [256, 256, 0, 0, True, 1, 4],
            [256, 256, 5, 0, True, 1, 2]
        ]
    },
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [256, 960, 1, 1],
            [960, 1280, 1, 1]
        ]
    }
}

MNV4ConvLarge_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 24, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [24, 48, 2, 4.0, True]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 96, 3, 5, True, 2, 4],
            [96, 96, 3, 3, True, 1, 4]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            [96, 192, 3, 5, True, 2, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 5, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 3, 0, True, 1, 4]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 13,
        "block_specs": [
            [192, 512, 5, 5, True, 2, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 3, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 3, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4]
        ]
    },
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [512, 960, 1, 1],
            [960, 1280, 1, 1]
        ]
    }
}


def mhsa(num_heads, key_dim, value_dim, px):
    if px == 24:
        kv_strides = 2
    elif px == 12:
        kv_strides = 1
    query_h_strides = 1
    query_w_strides = 1
    use_layer_scale = True
    use_multi_query = True
    use_residual = True
    return [
        num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides,
        use_layer_scale, use_multi_query, use_residual
    ]


MNV4HybridConvMedium_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [32, 48, 2, 4.0, True]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 80, 3, 5, True, 2, 4],
            [80, 80, 3, 3, True, 1, 2]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 8,
        "block_specs": [
            [80, 160, 3, 5, True, 2, 6],
            [160, 160, 0, 0, True, 1, 2],
            [160, 160, 3, 3, True, 1, 4],
            [160, 160, 3, 5, True, 1, 4, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 3, True, 1, 4, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 0, True, 1, 4, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 3, True, 1, 4, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 0, True, 1, 4]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 12,
        "block_specs": [
            [160, 256, 5, 5, True, 2, 6],
            [256, 256, 5, 5, True, 1, 4],
            [256, 256, 3, 5, True, 1, 4],
            [256, 256, 3, 5, True, 1, 4],
            [256, 256, 0, 0, True, 1, 2],
            [256, 256, 3, 5, True, 1, 2],
            [256, 256, 0, 0, True, 1, 2],
            [256, 256, 0, 0, True, 1, 4, mhsa(4, 64, 64, 12)],
            [256, 256, 3, 0, True, 1, 4, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 5, True, 1, 4, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 0, True, 1, 4, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 0, True, 1, 4]
        ]
    },
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [256, 960, 1, 1],
            [960, 1280, 1, 1]
        ]
    }
}

MNV4HybridConvLarge_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 24, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [24, 48, 2, 4.0, True]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 96, 3, 5, True, 2, 4],
            [96, 96, 3, 3, True, 1, 4]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            [96, 192, 3, 5, True, 2, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 5, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True, 1, 4, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True, 1, 4, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True, 1, 4, mhsa(8, 48, 48, 24)],
            [192, 192, 3, 0, True, 1, 4]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 14,
        "block_specs": [
            [192, 512, 5, 5, True, 2, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 3, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 3, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4]
        ]
    },
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [512, 960, 1, 1],
            [960, 1280, 1, 1]
        ]
    }
}

MODEL_SPECS = {
    "MobileNetV4ConvSmall": MNV4ConvSmall_BLOCK_SPECS,
    "MobileNetV4ConvMedium": MNV4ConvMedium_BLOCK_SPECS,
    "MobileNetV4ConvLarge": MNV4ConvLarge_BLOCK_SPECS,
    "MobileNetV4HybridMedium": MNV4HybridConvMedium_BLOCK_SPECS,
    "MobileNetV4HybridLarge": MNV4HybridConvLarge_BLOCK_SPECS
}


def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
    """
    This function is copied from here
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"
    This is to ensure that all layers have channels that are divisible by 8.
    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.
    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, act=False, squeeze_excitation=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=3, stride=stride))
        if squeeze_excitation:
            self.block.add_module('conv_3x3',
                                  conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 middle_dw_downsample,
                 stride,
                 expand_ratio
                 ):
        """An inverted bottleneck block with optional depthwises.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        """
        super().__init__()
        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)


    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        return x


class MultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(self, inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides,
                 dw_kernel_size=3, dropout=0.0):
        """Multi Query Attention with spatial downsampling.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        3 parameters are introduced for the spatial downsampling:
        1. kv_strides: downsampling factor on Key and Values only.
        2. query_h_strides: vertical strides on Query only.
        3. query_w_strides: horizontal strides on Query only.
        This is an optimized version.
        1. Projections in Attention is explict written out as 1x1 Conv2D.
        2. Additional reshapes are introduced to bring a up to 3x speed up.
        """
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        self.head_dim = key_dim // num_heads

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            self._query_downsampling_norm = nn.BatchNorm2d(inp)
        self._query_proj = conv_2d(inp, num_heads * key_dim, 1, 1, norm=False, act=False)

        if self.kv_strides > 1:
            self._key_dw_conv = conv_2d(inp, inp, dw_kernel_size, kv_strides, groups=inp, norm=True, act=False)
            self._value_dw_conv = conv_2d(inp, inp, dw_kernel_size, kv_strides, groups=inp, norm=True, act=False)
        self._key_proj = conv_2d(inp, key_dim, 1, 1, norm=False, act=False)
        self._value_proj = conv_2d(inp, key_dim, 1, 1, norm=False, act=False)

        self._output_proj = conv_2d(num_heads * key_dim, inp, 1, 1, norm=False, act=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, seq_length, _, _ = x.size()
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q = F.avg_pool2d(self.query_h_stride, self.query_w_stride)

            # q = F.avg_pool2d(q, kernel_size=(self.query_h_strides, self.query_w_strides))    #改了下尺寸
            q = self._query_downsampling_norm(q)
            q = self._query_proj(q)
        else:
            q = self._query_proj(x)
        px = q.size(2)
        q = q.view(batch_size, self.num_heads, -1, self.key_dim)  # [batch_size, num_heads, seq_length, key_dim]

        if self.kv_strides > 1:
            k = self._key_dw_conv(x)
            k = self._key_proj(k)
            v = self._value_dw_conv(x)
            v = self._value_proj(v)
        else:
            k = self._key_proj(x)
            v = self._value_proj(x)
        k = k.view(batch_size, self.key_dim, -1)  # [batch_size, key_dim, seq_length]
        v = v.view(batch_size, -1, self.key_dim)  # [batch_size, seq_length, key_dim]

        # calculate attn score
        attn_score = torch.matmul(q, k) / (self.head_dim ** 0.5)

        attn_score = self.dropout(attn_score)
        attn_score = F.softmax(attn_score, dim=-1)

        context = torch.matmul(attn_score, v)
        context = context.view(batch_size, self.num_heads * self.key_dim, px, px)
        output = self._output_proj(context)
        return output


class MNV4LayerScale(nn.Module):
    def __init__(self, init_value):
        """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        As used in MobileNetV4.
        Attributes:
            init_value (float): value to initialize the diagonal matrix of LayerScale.
        """
        super().__init__()
        self.init_value = init_value

    def forward(self, x):
        gamma = self.init_value * torch.ones(x.size(-1), dtype=x.dtype, device=x.device)
        return x * gamma


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(
            self,
            inp,
            num_heads,
            key_dim,
            value_dim,
            query_h_strides,
            query_w_strides,
            kv_strides,
            use_layer_scale,
            use_multi_query,
            use_residual=True
    ):
        super().__init__()
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.use_layer_scale = use_layer_scale
        self.use_multi_query = use_multi_query
        self.use_residual = use_residual

        self._input_norm = nn.BatchNorm2d(inp)
        if self.use_multi_query:
            self.multi_query_attention = MultiQueryAttentionLayerWithDownSampling(
                inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides
            )
        else:
            self.multi_head_attention = nn.MultiheadAttention(inp, num_heads, kdim=key_dim)

        if self.use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale = MNV4LayerScale(self.layer_scale_init_value)

    def forward(self, x):
        # Not using CPE, skipped
        # input norm
        shortcut = x
        x = self._input_norm(x)
        # multi query
        if self.use_multi_query:
            x = self.multi_query_attention(x)
        else:
            x = self.multi_head_attention(x, x)
        # layer scale
        if self.use_layer_scale:
            x = self.layer_scale(x)
        # use residual
        if self.use_residual:
            x = x + shortcut
        return x


def build_blocks(layer_spec):
    if not layer_spec.get('block_name'):
        return nn.Sequential()
    block_names = layer_spec['block_name']
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ['inp', 'oup', 'kernel_size', 'stride']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"convbn_{i}", conv_2d(**args))
    elif block_names == "uib":
        schema_ = ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride',
                   'expand_ratio', 'msha']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            msha = args.pop("msha") if "msha" in args else 0
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
            if msha:
                msha_schema_ = [
                    "inp", "num_heads", "key_dim", "value_dim", "query_h_strides", "query_w_strides", "kv_strides",
                    "use_layer_scale", "use_multi_query", "use_residual"
                ]
                args = dict(zip(msha_schema_, [args['oup']] + (msha)))
                layers.add_module(f"msha_{i}", MultiHeadSelfAttentionBlock(**args))
    elif block_names == "fused_ib":
        schema_ = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers


class MobileNetV4(nn.Module):
    def __init__(self, model):
        # MobileNetV4ConvSmall  MobileNetV4ConvMedium  MobileNetV4ConvLarge
        # MobileNetV4HybridMedium  MobileNetV4HybridLarge
        """Params to initiate MobilenNetV4
        Args:
            model : support 5 types of models as indicated in
            "https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py"
        """
        super().__init__()
        assert model in MODEL_SPECS.keys()
        self.model = model
        self.spec = MODEL_SPECS[self.model]

        # conv0
        self.conv0 = build_blocks(self.spec['conv0'])
        # layer1
        self.layer1 = build_blocks(self.spec['layer1'])
        # layer2
        self.layer2 = build_blocks(self.spec['layer2'])
        # layer3
        self.layer3 = build_blocks(self.spec['layer3'])
        # layer4
        self.layer4 = build_blocks(self.spec['layer4'])
        # layer5
        self.layer5 = build_blocks(self.spec['layer5'])
        self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # x5 = self.layer5(x4)
        # x5 = nn.functional.adaptive_avg_pool2d(x5, 1)
        return [x1, x2, x3, x4]


###################################
def MobileNetV4ConvSmall():
    model = MobileNetV4('MobileNetV4ConvSmall')
    return model


def MobileNetV4ConvMedium():  # 以前训练都是用的这个
    model = MobileNetV4('MobileNetV4ConvMedium')
    return model


def MobileNetV4ConvLarge():
    model = MobileNetV4('MobileNetV4ConvLarge')
    return model


#################################################
def MobileNetV4HybridMedium():
    model = MobileNetV4('MobileNetV4HybridMedium')
    return model


def MobileNetV4HybridLarge():
    model = MobileNetV4('MobileNetV4HybridLarge')
    return model

    # Model




class AtrousSeparableConvolution(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      bias=bias, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=bias),
        )

    def forward(self, x):
        return self.body(x)


class AAPM(nn.Module):
    def __init__(self, in_channels, dilation_rates):
        super().__init__()

        up_channels = int(in_channels * 2.5)
        self.num_branches = len(dilation_rates)

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, up_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(up_channels),
            nn.SiLU()
        )

        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(up_channels, up_channels, kernel_size=3, padding=rate,
                              dilation=rate, groups=up_channels, bias=False),
                    nn.BatchNorm2d(up_channels),
                    nn.SiLU()
                )
            )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 缩减维度以减少参数
        d = max(int(up_channels / 4), 16)
        self.fc1 = nn.Sequential(
            nn.Conv2d(up_channels, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.SiLU()
        )
        self.fc2 = nn.Conv2d(d, up_channels * self.num_branches, kernel_size=1, bias=False)
        self.down_channels = nn.Conv2d(up_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_up = self.expand(x)

        branch_outs = [branch(x_up) for branch in self.branches]

        U = sum(branch_outs)

        S = self.global_pool(U)
        Z = self.fc1(S)
        A = self.fc2(Z)

        A = A.view(x.size(0), self.num_branches, x_up.size(1), 1, 1)
        A = self.softmax(A)

        V = torch.stack(branch_outs, dim=1)  # 形状: [B, num_branches, up_channels, H, W]
        V = (V * A).sum(dim=1)  # 按照 num_branches 维度加权求和，降回 [B, up_channels, H, W]

        V = self.down_channels(V)
        return V + x


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class EdgeEnhancer(nn.Module):


    def __init__(self, in_channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.edge_weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        smooth = self.max_pool(x)
        high_freq = smooth - x

        weights = self.edge_weight(high_freq)

        return x + high_freq * weights


class FCF_UpBlock(nn.Module):
    """
    Progressive Frequency-Calibration Fusion Block
    替代传统的 Up + Concat
    """

    def __init__(self, in_channels_deep, in_channels_skip, out_c):
        super().__init__()

        self.up = DySample(in_channels_deep, 2)
        self.deep_proj = nn.Sequential(
            nn.Conv2d(in_channels_deep, in_channels_skip, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels_skip),
            nn.SiLU()
        )

        self.edge_enhancer = EdgeEnhancer(in_channels_skip)

        self.calibration = nn.Sequential(
            nn.Conv2d(in_channels_skip * 2, in_channels_skip, kernel_size=1),  # 融合两者
            nn.BatchNorm2d(in_channels_skip),
            nn.Sigmoid()  # 生成 0-1 的门控权重
        )

        self.final_conv = AtrousSeparableConvolution(in_channels_skip * 2,
                                                     out_c, kernel_size=3, padding=1)

    def forward(self, x_deep, x_skip):

        x_up = self.up(x_deep)

        x_up_proj = self.deep_proj(x_up)  # [B, C_skip, H, W]

        x_skip_edge = self.edge_enhancer(x_skip)

        cat_features = torch.cat([x_up_proj, x_skip_edge], dim=1)
        calib_gate = self.calibration(cat_features)

        x_skip_calibrated = x_skip_edge * calib_gate
        x_out = torch.cat([x_up_proj, x_skip_calibrated], dim=1)

        return self.final_conv(x_out)


####################################################################################################################


class LSN_decoder(nn.Module):
    def __init__(self, low_channels, high_channels, num_classes, isDSV=False):
        super(LSN_decoder, self).__init__()

        self.project = nn.Sequential(
            nn.Conv2d(low_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.DSME = AAPM(high_channels, [1, 2, 3])  # 128
        self.FCF = FCF_UpBlock(high_channels, 48, high_channels + 48)
        ##########################################################################################
        # self.DySample1 = DySample(high_channels, scale=2)
        self.DySample2 = DySample(high_channels + 48, scale=4)
        self.DySample3 = DySample(num_classes, scale=4, groups=2)

        ##########################################################################################

        self.classifier = nn.Sequential(
            nn.Conv2d(high_channels + 48, high_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_channels, num_classes, 1)
        )
        # 是否使用深监督
        self.DSV_conv = nn.Conv2d(48, num_classes, 1)

    def forward(self, low_feature, high_feature, isDSV=False):
        low_level_feature = self.project(low_feature)

        AAPM_feature = self.DSME(high_feature)
        ##########################################################################################

        concat_ = self.FCF(AAPM_feature, low_level_feature)

        concat_up = self.DySample2(concat_)

        output_feature = self.classifier(concat_up)

        DySample_concat_ = self.DySample3(output_feature)
        DSV_low_feature = self.DSV_conv(low_level_feature)

        return [DSV_low_feature, DySample_concat_]


def _segm_mobilenetv4(num_classes, pretrained_backbone):
    backbone = MobileNetV4ConvSmall()
    # [32, 64, 96, 128]
    low_channels = 96
    high_channels = 128
    decoder = LSN_decoder(low_channels, high_channels, num_classes)
    model = _BaseSegmentationModel(backbone, decoder)
    return model


class _BaseSegmentationModel(nn.Module):
    def __init__(self, backbone, decoder):
        super(_BaseSegmentationModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        features = self.backbone(x)
        low_features = features[2]
        high_features = features[3]
        out = self.decoder(low_features, high_features)
        return out


def LSN_origin(num_classes, pretrained):
    return _segm_mobilenetv4(num_classes, pretrained)

