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

        # TODO
        self.analyze = False
        self.an_x = None
        self.an_y = None

        nn.init.ones_(self.weight)  # gamma (weight)
        nn.init.zeros_(self.bias)   # beta (bias)

    def forward(self, x):
        y = super(BatchNorm2d, self).forward(x)
        # TODO
        if self.analyze:
            self.an_x = x
            self.an_y = y
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

        # TODO
        self.analyze = False
        self.an_w = None
        self.an_x = None
        self.an_y = None
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
        if q_scheme == 'sat':
            if q_bits_w is not None: self.fake_quantizer_w = SATWeightFakeQuantizer(q_bits_w)
            if q_bits_a is not None: self.fake_quantizer_a = SATActFakeQuantizer(q_bits_a, 8.0)
        elif q_scheme == 'lsq':
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
            # TODO
            if self.analyze:
                self.an_x_fq = x_used
        else:
            x_used = x

        if self.q_bits_w is not None:
            w_used = self.fake_quantizer_w(self.conv.weight)    # w_fq
            # TODO
            if self.analyze:
                self.an_w_fq = w_used
        else:
            w_used = self.conv.weight

        y = self._forward_conv(x_used, w_used, self.conv.bias)
        # TODO
        if self.analyze:
            self.an_x = x
            self.an_w = self.conv.weight.data
            self.an_y = y
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

        # TODO
        self.analyze = False
        self.an_w = None
        self.an_x = None
        self.an_y = None
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
        if q_scheme == 'sat':
            if q_bits_w is not None:
                if last_layer: 
                    # Turn scale adjustment on at the last FC layer weight quantization
                    self.fake_quantizer_w = SATWeightFakeQuantizer(q_bits_w, out_features=out_features)
                else:
                    self.fake_quantizer_w = SATWeightFakeQuantizer(q_bits_w)
            if q_bits_a is not None:
                self.fake_quantizer_a = SATActFakeQuantizer(q_bits_a, 10.0)
        elif q_scheme == 'lsq':
            if q_bits_w is not None: self.fake_quantizer_w = LSQWeightFakeQuantizer(q_bits_w)
            # if q_bits_w is not None: self.fake_quantizer_w = LSQWeightFakeQuantizer_constant_rescale(q_bits_w, out_features=out_features) # only for comparison to SAT!!!
            if q_bits_a is not None: self.fake_quantizer_a = LSQActFakeQuantizer(q_bits_a)
        else:
            assert q_scheme is None
        self.last_layer = last_layer

    def forward(self, x):
        if self.q_bits_a is not None:
            x_used = self.fake_quantizer_a(x)   # x_fq
            # TODO
            if self.analyze:
                self.an_x_fq = x_used                
        else:
            x_used = x

        if self.q_bits_w is not None:
            w_used = self.fake_quantizer_w(self.fc.weight)    # w_fq
            # TODO
            if self.analyze:
                self.an_w_fq = w_used
        else:
            w_used = self.fc.weight

        if (self.q_scheme == 'sat') and (self.q_bits_w is not None) and self.last_layer and (not self.training) and (self.fc.bias is not None):
            # Refer to https://github.com/deJQK/AdaBits/blob/master/models/quant_ops.py#L274
            bias_used = self.fc.bias / self.fake_quantizer_w.weight_scale
        else:
            bias_used = self.fc.bias

        y = F.linear(x_used, w_used, bias=bias_used)
        # TODO
        if self.analyze:
            self.an_x = x
            self.an_w = self.fc.weight.data
            self.an_b = bias_used.data
            self.an_y = y
        return y

class QLinear_2(nn.Module):
    def __init__(
        self,
        q_scheme,
        q_bits_w,
        q_bits_a,
        in_features, 
        out_features, 
        bias=True,
        last_layer=False):
        super(QLinear_2, self).__init__()

        # TODO
        self.analyze = False
        self.an_w = None
        self.an_x = None
        self.an_y = None
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
        if q_scheme == 'sat':
            if q_bits_w is not None:
                if last_layer: 
                    # Turn scale adjustment on at the last FC layer weight quantization
                    self.fake_quantizer_w = SATWeightFakeQuantizer(q_bits_w)  # NOT USE WEIGHT RESCALE
                else:
                    self.fake_quantizer_w = SATWeightFakeQuantizer(q_bits_w)
            if q_bits_a is not None:
                self.fake_quantizer_a = SATActFakeQuantizer(q_bits_a, 10.0)
        elif q_scheme == 'lsq':
            if q_bits_w is not None: self.fake_quantizer_w = LSQWeightFakeQuantizer(q_bits_w)
            # if q_bits_w is not None: self.fake_quantizer_w = LSQWeightFakeQuantizer_constant_rescale(q_bits_w, out_features=out_features) # only for comparison to SAT!!!
            if q_bits_a is not None: self.fake_quantizer_a = LSQActFakeQuantizer(q_bits_a)
        else:
            assert q_scheme is None
        self.last_layer = last_layer

    def forward(self, x):
        if self.q_bits_a is not None:
            x_used = self.fake_quantizer_a(x)   # x_fq
            # TODO
            if self.analyze:
                self.an_x_fq = x_used                
        else:
            x_used = x

        if self.q_bits_w is not None:
            w_used = self.fake_quantizer_w(self.fc.weight)    # w_fq
            # TODO
            if self.analyze:
                self.an_w_fq = w_used
        else:
            w_used = self.fc.weight

        if (self.q_scheme == 'sat') and (self.q_bits_w is not None) and self.last_layer and (not self.training) and (self.fc.bias is not None):
            # Refer to https://github.com/deJQK/AdaBits/blob/master/models/quant_ops.py#L274
            bias_used = self.fc.bias / self.fake_quantizer_w.weight_scale
        else:
            bias_used = self.fc.bias

        y = F.linear(x_used, w_used, bias=bias_used)
        # TODO
        if self.analyze:
            self.an_x = x
            self.an_w = self.fc.weight.data
            self.an_b = bias_used.data
            self.an_y = y
        return y


"""
Composite module
"""
# Refer to https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L59 and https://github.com/zhutmost/lsq-net/blob/master/model/resnet.py#L33
class QBasicBlock(nn.Module):
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
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None):
        super(QBasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups = 1 and base_width = 64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')

        if (q_bits_w is None) and (q_bits_a is None): q_scheme = None

        self.conv1 = QConv2d(
            q_scheme, q_bits_w, q_bits_a,
            in_planes, planes, 3,
            stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes, track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QConv2d(
            q_scheme, q_bits_w, q_bits_a,
            planes, planes, 3,
            padding=1, bias=False)
        self.bn2 = norm_layer(planes, track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample is not None:
            residual = self.downsample(x)

        y += residual
        y = self.relu(y)
        return y

# Refer to https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L108 and https://github.com/zhutmost/lsq-net/blob/master/model/resnet.py#L73
class QBottleneck(nn.Module):
    expansion = 4

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
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None):
        super(QBottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        if (q_bits_w is None) and (q_bits_a is None): q_scheme = None

        self.conv1 = QConv2d(
            q_scheme, q_bits_w, q_bits_a,
            in_planes, width, 1,
            bias=False)
        self.bn1 = norm_layer(width, track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QConv2d(
            q_scheme, q_bits_w, q_bits_a,
            width, width, 3,
            stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
        self.bn2 = norm_layer(width, track_running_stats)
        self.conv3 = QConv2d(
            q_scheme, q_bits_w, q_bits_a,
            width, planes * self.expansion, 1,
            bias=False)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)

        if self.downsample is not None:
            residual = self.downsample(x)

        y += residual
        y = self.relu(y)
        return y

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

# Refer to https://github.com/soeaver/pytorch-priv/blob/master/models/imagenet/preresnet.py#L57
class QPreactBottleneck(nn.Module):
    expansion = 4

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
        super(QPreactBottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d
        self.downsample = downsample
        self.stride = stride        

        if (q_bits_w is None) and (q_bits_a is None): q_scheme = None

        self.bn1 = norm_layer(in_planes, track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = QConv2d(
            q_scheme, q_bits_w, q_bits_a,
            in_planes, planes, 1, 
            bias=False)
        self.bn2 = norm_layer(planes, track_running_stats)
        self.conv2 = QConv2d(
            q_scheme, q_bits_w, q_bits_a,
            planes, planes, 3, 
            stride=stride, padding=1, bias=False)
        self.bn3 = norm_layer(planes, track_running_stats)
        self.conv3 = QConv2d(
            q_scheme, q_bits_w, q_bits_a,
            planes, planes * self.expansion, 1, 
            bias=False)

    def forward(self, x):
        y = self.bn1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn3(y)
        y = self.relu(y)
        y = self.conv3(y)

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
# Refer to https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L166 and https://github.com/zhutmost/lsq-net/blob/master/model/resnet.py#L122

class QResNet(nn.Module):
    def __init__(
        self,
        q_scheme,
        q_bits_w,
        q_bits_a,
        track_running_stats,
        block,
        num_blocks,
        num_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        weight_init='kaiming'):
        super(QResNet, self).__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer
        self.track_running_stats = track_running_stats
        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None '
                f'or a 3-element tuple, got {replace_stride_with_dilation}')
        self.groups = groups
        self.base_width = width_per_group

        if (q_bits_w is None) and (q_bits_a is None): q_scheme = None
        self.q_scheme = q_scheme
        self.q_bits_w = q_bits_w
        self.q_bits_a = q_bits_a
        q_bits_w_first, q_bits_a_first, q_bits_w_last, q_bits_a_last = get_extra_q_bits(q_scheme, q_bits_w, q_bits_a)

        self.conv1 = QConv2d(
            q_scheme, q_bits_w_first, q_bits_a_first,
            3, self.in_planes, 7, 
            stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes, track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = QLinear(
            q_scheme, q_bits_w_last, q_bits_a_last,
            512 * block.expansion, num_classes,
            last_layer=True)

        if weight_init is not None:
            for m in self.modules():
                if isinstance(m, QConv2d):
                    if weight_init == 'kaiming':
                        nn.init.kaiming_normal_(m.conv.weight, mode='fan_out', nonlinearity='relu')
                    elif weight_init == 'xavier':
                        nn.init.xavier_normal_(m.conv.weight)
                if isinstance(m, (BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, QBottleneck) and (m.bn3.weight is not None):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, QBasicBlock) and (m.bn2.weight is not None):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride=1,
        dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if (stride != 1) or (self.in_planes != planes * block.expansion):
            downsample = nn.Sequential(
                QConv2d(
                    self.q_scheme, self.q_bits_w, self.q_bits_a,
                    self.in_planes, planes * block.expansion, 1,
                    stride=stride, bias=False),
                norm_layer(planes * block.expansion, self.track_running_stats))

        layers = []
        layers.append(
            block(
                self.q_scheme, self.q_bits_w, self.q_bits_a, self.track_running_stats,
                self.in_planes, planes, 
                stride=stride, downsample=downsample, groups=self.groups, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.q_scheme, self.q_bits_w, self.q_bits_a, self.track_running_stats,
                    self.in_planes, planes,
                    groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.max_pool(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avg_pool(y)
        y = torch.flatten(y, 1)
        self.backbone_out = y.clone().detach()  # For LAME
        y = self.fc(y)
        return y

# Models
def q_resnet18(
    pretrained,
    q_scheme,
    q_bits_w,
    q_bits_a,
    track_running_stats,
    num_classes,
    source_dataset,
    weight_init):
    model = QResNet(
        q_scheme, q_bits_w, q_bits_a, track_running_stats,
        QBasicBlock, [2, 2, 2, 2], num_classes,
        weight_init)
    if pretrained:
        assert source_dataset == 'imagenet'
        pretrained_state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')  
        model.load_state_dict(pretrained_state_dict, strict=False)
        for m_name, m in model.named_modules():
            if type(m) == QConv2d:
                m.conv.weight.data = pretrained_state_dict[m_name + '.weight']
                if m.conv.bias is not None: m.conv.bias.data = pretrained_state_dict[m_name + '.bias'] 
            elif type(m) == QLinear:
                m.fc.weight.data = pretrained_state_dict[m_name + '.weight']
                if m.fc.bias is not None: m.fc.bias.data = pretrained_state_dict[m_name + '.bias'] 
    return model

def q_resnet50(
    pretrained,
    q_scheme,
    q_bits_w,
    q_bits_a,
    track_running_stats,
    num_classes,
    source_dataset,
    weight_init):
    model = QResNet(
        q_scheme, q_bits_w, q_bits_a, track_running_stats,
        QBottleneck, [3, 4, 6, 3], num_classes,
        weight_init)
    if pretrained:
        if source_dataset == 'imagenet':
            assert source_dataset == 'imagenet'
            pretrained_state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-0676ba61.pth')  
            model.load_state_dict(pretrained_state_dict, strict=False)
            for m_name, m in model.named_modules():
                if type(m) == QConv2d:
                    m.conv.weight.data = pretrained_state_dict[m_name + '.weight']
                    if m.conv.bias is not None: m.conv.bias.data = pretrained_state_dict[m_name + '.bias'] 
                elif type(m) == QLinear:
                    m.fc.weight.data = pretrained_state_dict[m_name + '.weight']
                    if m.fc.bias is not None: m.fc.bias.data = pretrained_state_dict[m_name + '.bias'] 
        else:
            assert source_dataset == 'cifar10'
            pretrained_ckpt = torch.load(f'./checkpoint/resnet50_{source_dataset}.pt')
            model.load_state_dict(pretrained_ckpt['model'], strict=True)
    return model

# Refer to https://github.com/soeaver/pytorch-priv/blob/master/models/imagenet/preresnet.py#L95 and https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/preresnet.py#L273
class QPreactResNet(nn.Module):
    def __init__(
        self,
        q_scheme,
        q_bits_w,
        q_bits_a,
        track_running_stats,
        block, 
        num_blocks, 
        num_classes,
        norm_layer=None):
        super(QPreactResNet, self).__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer
        self.track_running_stats = track_running_stats
        self.in_planes = 64

        if (q_bits_w is None) and (q_bits_a is None): q_scheme = None
        self.q_scheme = q_scheme
        self.q_bits_w = q_bits_w
        self.q_bits_a = q_bits_a
        q_bits_w_first, q_bits_a_first, q_bits_w_last, q_bits_a_last = get_extra_q_bits(q_scheme, q_bits_w, q_bits_a)

        self.conv1 = QConv2d(
            q_scheme, q_bits_w_first, q_bits_a_first,
            3, self.in_planes, 7,
            stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes, track_running_stats)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn_final = norm_layer(512 * block.expansion, track_running_stats)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = QLinear(
            q_scheme, q_bits_w_last, q_bits_a_last,            
            512 * block.expansion, num_classes,
            last_layer=True)

        for m in self.modules():
            if isinstance(m, QConv2d):
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
        if (stride != 1) or (self.in_planes != planes * block.expansion):
            downsample = nn.Sequential(
                QConv2d(
                    self.q_scheme, self.q_bits_w, self.q_bits_a,
                    self.in_planes, planes * block.expansion, 1, 
                    stride=stride, bias=False),
                norm_layer(planes * block.expansion, self.track_running_stats))

        layers = []
        layers.append(block(
            self.q_scheme, self.q_bits_w, self.q_bits_a, self.track_running_stats,
            self.in_planes, planes, 
            stride=stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(
                self.q_scheme, self.q_bits_w, self.q_bits_a, self.track_running_stats,
                self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.max_pool(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.bn_final(y)
        y = self.relu2(y)
        y = self.avg_pool(y)
        y = torch.flatten(y, 1)
        self.backbone_out = y.clone().detach()  # For LAME
        y = self.fc(y)
        return y

def q_preact_resnet18(
    pretrained,
    q_scheme,
    q_bits_w,
    q_bits_a,
    track_running_stats,
    num_classes,
    source_dataset):
    model = QPreactResNet(
        q_scheme, q_bits_w, q_bits_a, track_running_stats,
        QPreactBasicBlock, [2, 2, 2, 2], num_classes)
    if pretrained:
        assert source_dataset == 'imagenet'
        pretrained_ckpt = torch.load(f'./pretrained_checkpoint/preact_resnet18_on_{source_dataset}.pt')
        model.load_state_dict(pretrained_ckpt['model'], strict=False)
    return model

def q_preact_resnet50(
    pretrained,
    q_scheme,
    q_bits_w,
    q_bits_a,
    track_running_stats,
    num_classes,
    source_dataset):
    model = QPreactResNet(
        q_scheme, q_bits_w, q_bits_a, track_running_stats,
        QPreactBottleneck, [3, 4, 6, 3], num_classes)
    if pretrained:
        assert source_dataset == 'imagenet'
        pretrained_ckpt = torch.load(f'./pretrained_checkpoint/preact_resnet50_on_{source_dataset}.pt')
        model.load_state_dict(pretrained_ckpt['model'], strict=False)
    return model

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

class QPreactResNetForCIFAR_2(nn.Module):   # For CIFAR or SVHN
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
        super(QPreactResNetForCIFAR_2, self).__init__()

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
        self.fc = QLinear_2(
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
    num_classes,
    source_dataset,
    weight_init):
    model = QPreactResNetForCIFAR(
        q_scheme, q_bits_w, q_bits_a, track_running_stats,
        26, num_classes)
    if pretrained:
        assert source_dataset in ['cifar10', 'svhn']
        pretrained_ckpt = torch.load(f'./pretrained_checkpoint/preact_resnet26_on_{source_dataset}.pt')
        model.load_state_dict(pretrained_ckpt['model'], strict=False)
    return model
