import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant import LSQWeightFakeQuantizer, LSQActFakeQuantizer, get_extra_q_bits

"""
Basic module
"""
class BatchNorm2d(nn.BatchNorm2d):          # nn.BatchNorm2d with affine=True. Not quantized.
    def __init__(
        self,
        num_features,
        track_running_stats):
        super(BatchNorm2d, self).__init__(
            num_features,
            track_running_stats=track_running_stats)

        nn.init.ones_(self.weight)  # gamma (weight)
        nn.init.zeros_(self.bias)   # beta (bias)

    def forward(self, x):
        y = super(BatchNorm2d, self).forward(x)
        return y

class QConv2d(nn.Module):
    def __init__(
        self,
        q_scheme,               # Quantization(-aware training) scheme. Meaningless if (q_bits_w is None) and (q_bits_a is None).
        q_bits_w,               # Bitwidth of conv weight after quantized. It can be different to q_bits_w defined in the model.
        q_bits_a,               # Bitwidth of conv input (regarded as activation) after quantized. It can be different to q_bits_a defined in the model.
        in_channels,
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode='zeros'):
        super(QConv2d, self).__init__()

        if q_bits_w is not None:
            self.an_w_fq = None
        # bias is set to False
        if q_bits_a is not None:
            self.an_x_fq = None

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.conv.reset_parameters()

        if (q_bits_w is None) and (q_bits_a is None): q_scheme = None
        self.q_bits_w = q_bits_w
        self.q_bits_a = q_bits_a
        if q_scheme == 'lsq':
            if q_bits_w is not None: self.fake_quantizer_w = LSQWeightFakeQuantizer(q_bits_w)
            if q_bits_a is not None: self.fake_quantizer_a = LSQActFakeQuantizer(q_bits_a)
        else:
            assert q_scheme is None

    # Refer to https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/quantization/scalar/modules/qconv.py#L75
    def _forward_conv(self, x, w, bias):
        if self.conv.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(x, self.conv._padding_repeated_twice, mode=self.conv.padding_mode), w,
                bias=bias, stride=self.conv.stride, padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
        else:
            return F.conv2d(
                x, w,
                bias=bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)

    def forward(self, x):
        if self.q_bits_a is not None:
            x_used = self.fake_quantizer_a(x)   # x_fq
        else:
            x_used = x

        if self.q_bits_w is not None:
            w_used = self.fake_quantizer_w(self.conv.weight)    # w_fq
        else:
            w_used = self.conv.weight

        y = self._forward_conv(x_used, w_used, self.conv.bias)

        return y

class QLinear(nn.Module):
    def __init__(
        self,
        q_scheme,
        q_bits_w,
        q_bits_a,
        in_features, 
        out_features, 
        bias=True,
        last_layer=False):
        super(QLinear, self).__init__()

        if q_bits_w is not None:
            self.an_w_fq = None
        if bias:
            self.an_b = None
        if q_bits_a is not None:
            self.an_x_fq = None

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.fc.reset_parameters()

        if (q_bits_w is None) and (q_bits_a is None): q_scheme = None
        self.q_scheme = q_scheme
        self.q_bits_w = q_bits_w
        self.q_bits_a = q_bits_a
        if q_scheme == 'lsq':
            if q_bits_w is not None: self.fake_quantizer_w = LSQWeightFakeQuantizer(q_bits_w)
            if q_bits_a is not None: self.fake_quantizer_a = LSQActFakeQuantizer(q_bits_a)
        else:
            assert q_scheme is None
        self.last_layer = last_layer

    def forward(self, x):
        if self.q_bits_a is not None:
            x_used = self.fake_quantizer_a(x)   # x_fq      
        else:
            x_used = x

        if self.q_bits_w is not None:
            w_used = self.fake_quantizer_w(self.fc.weight)    # w_fq
        else:
            w_used = self.fc.weight

        y = F.linear(x_used, w_used, bias=self.fc.bias)

        return y


"""
Composite module
"""
# Refer to https://github.com/soeaver/pytorch-priv/blob/master/models/imagenet/preresnet.py#L26 and https://github.com/jmiemirza/DUA/blob/master/models/resnet_26.py#L10
class QPreactBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, 
        q_scheme,
        q_bits_w,
        q_bits_a,
        track_running_stats,
        in_planes, 
        planes,
        stride=1,
        downsample=None,
        norm_layer=None):
        super(QPreactBasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d
        self.downsample = downsample
        self.stride = stride        

        if (q_bits_w is None) and (q_bits_a is None): q_scheme = None

        self.bn1 = norm_layer(in_planes, track_running_stats)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = QConv2d(
            q_scheme, q_bits_w, q_bits_a,
            in_planes, planes, 3, 
            stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes, track_running_stats)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = QConv2d(
            q_scheme, q_bits_w, q_bits_a,
            planes, planes, 3, 
            padding=1, bias=False)

    def forward(self, x):
        y = self.bn1(x)
        y = self.relu1(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.conv2(y)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        y = y + residual
        return y


# Refer to https://github.com/jmiemirza/DUA/blob/master/models/resnet_26.py#L39
class Downsample(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super(Downsample, self).__init__()

        self.avg_pool = nn.AvgPool2d(stride)
        assert planes % in_planes == 0
        self.expand_ratio = planes // in_planes

    def forward(self, x):
        x = self.avg_pool(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)

"""
Model
"""

# Refer to https://github.com/jmiemirza/DUA/blob/master/models/resnet_26.py#L51
class QPreactResNetForCIFAR(nn.Module):   # For CIFAR or SVHN
    def __init__(
        self,
        q_scheme,
        q_bits_w,
        q_bits_a,
        track_running_stats,
        depth,
        num_classes,
        width=1,
        norm_layer=None,
        weight_init='kaiming'):
        super(QPreactResNetForCIFAR, self).__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer
        self.track_running_stats = track_running_stats
        self.in_planes = 16
        block = QPreactBasicBlock
        assert (depth - 2) % 6 == 0
        num_blocks = (depth - 2) // 6

        if (q_bits_w is None) and (q_bits_a is None): q_scheme = None
        self.q_scheme = q_scheme
        self.q_bits_w = q_bits_w
        self.q_bits_a = q_bits_a
        q_bits_w_first, q_bits_a_first, q_bits_w_last, q_bits_a_last = get_extra_q_bits(q_scheme, q_bits_w, q_bits_a)

        self.conv1 = QConv2d(
            q_scheme, q_bits_w_first, q_bits_a_first,
            3, self.in_planes, 3, 
            padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16 * width, num_blocks)
        self.layer2 = self._make_layer(block, 32 * width, num_blocks, stride=2)
        self.layer3 = self._make_layer(block, 64 * width, num_blocks, stride=2)
        self.bn_final = norm_layer(64 * width, track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = QLinear(
            q_scheme, q_bits_w_last, q_bits_a_last,            
            64 * width, num_classes,
            last_layer=True)

        if weight_init is not None:
            for m in self.modules():
                if isinstance(m, QConv2d):
                    if weight_init == 'kaiming':
                        nn.init.kaiming_normal_(m.conv.weight, mode='fan_out', nonlinearity='relu')
                    elif weight_init == 'xavier':
                        nn.init.xavier_normal_(m.conv.weight)
                    elif weight_init == 'normal':
                        n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                        m.conv.weight.data.normal_(0, math.sqrt(2. / n))
                if isinstance(m, (BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, 
        block, 
        planes, 
        num_blocks, 
        stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if (stride != 1) or (self.in_planes != planes):
            downsample = Downsample(self.in_planes, planes, stride)

        layers = [block(
            self.q_scheme, self.q_bits_w, self.q_bits_a, self.track_running_stats,
            self.in_planes, planes,
            stride=stride, downsample=downsample)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(
                self.q_scheme, self.q_bits_w, self.q_bits_a, self.track_running_stats,
                self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.bn_final(y)
        y = self.relu(y)
        y = self.avg_pool(y)
        y = torch.flatten(y, 1)
        self.backbone_out = y.clone().detach()  # For LAME
        y = self.fc(y)
        return y


def q_preact_resnet26(
    pretrained,
    q_scheme,
    q_bits_w,
    q_bits_a,
    track_running_stats,
    num_classes):
    model = QPreactResNetForCIFAR(
        q_scheme, q_bits_w, q_bits_a, track_running_stats,
        26, num_classes)
    if pretrained:
        pretrained_ckpt = torch.load(f'./preact_resnet26_on_cifar10.pt')
        model.load_state_dict(pretrained_ckpt['model'], strict=False)
    return model
