import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
import torch.nn.functional as F
import pdb
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

from share import share_dict
from .build import BACKBONE_REGISTRY
from .backbone import Backbone


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GradReverse(Function):
    # @staticmethod
    # def __init__(self, lambd):
    #    self.lambd = lambd
    @staticmethod
    def forward(grev, x, lmbd):
        # self.lmbd = lmbd
        grev.lmbd = lmbd
        return x.view_as(x)

    @staticmethod
    def backward(grev, grad_output):
        # print (grev.lmbd)
        return (grad_output * -grev.lmbd), None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class ResNetDRA(Backbone):

    def __init__(self, block, layers,
                 ms_class=None,
                 ms_type='random',
                 ms_layers=[],
                 ms_p=0.5,
                 ms_a=0.1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, pretrain=None):
        super(ResNetDRA, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._out_features = 512 * block.expansion

        self.mixstyle = None
        if ms_layers:
            self.mixstyle = ms_class(p=ms_p, alpha=ms_a, mix=ms_type)
            for layer_name in ms_layers:
                assert layer_name in ['layer1', 'layer2', 'layer3']
            print(f'Insert MixStyle after {ms_layers}')
        self.ms_layers = ms_layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.constant_(m.weight, 1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.new_param, self.init_param = [], []
        if pretrain:
            print('loading pretrain model from %s' % (pretrain))
            model = torch.load(pretrain)  # ['state_dict']
            if 'state_dict' in model:
                model = model['state_dict']

            prefixs = ['', 'module.features.', 'module.G.']
            new_params = self.state_dict().copy()
            for x in new_params:
                flag = 0
                for prefix in prefixs:
                    if prefix + x in model:
                        new_params[x] = model[prefix + x]
                        self.init_param.append(x)
                        flag = 1
                        break
                if not flag:
                    self.new_param.append(x)
                    print(x)
            self.load_state_dict(new_params)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if 'layer1' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer2(x)
        if 'layer2' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer3(x)
        if 'layer3' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockKDRN(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=None) -> None:
        super(BasicBlockKDRN, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        inp, oup = inplanes, planes * 4
        self.squeeze = inp // 16
        self.dim = int(math.sqrt(inp))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp, self.squeeze, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.squeeze, 4, bias=False),
        )
        self.sf = nn.Softmax(dim=1)
        self.conv_33 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, groups=planes, bias=False)
        self.conv_11 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_31 = nn.Conv2d(planes, planes, kernel_size=(3, 1), stride=stride, padding=[1, 0], bias=False)
        self.conv_13 = nn.Conv2d(planes, planes, kernel_size=(1, 3), stride=stride, padding=[0, 1], bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, 4, 1, 1, 1)
        y = self.sf(y)

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        dyres = self.conv_33(out) * y[:, 0] + self.conv_11(out) * y[:, 1] + \
                self.conv_31(out) * y[:, 2] + self.conv_13(out) * y[:, 3]

        out = dyres + self.conv2(out)
        # out = self.conv2(out)
        # out = dyres + out

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

@BACKBONE_REGISTRY.register(nick_name="Block_v4")
class BottleneckKDRA(nn.Module):
    expansion = 4
    fix_who = None
    noise_rate = 0.01

    @classmethod
    def fix(cls, fix_who, noise_rate=0.01):
        assert fix_who in ['meta', 'conv']
        cls.fix_who = fix_who
        cls.noise_rate = noise_rate

    @classmethod
    def unfix(cls):
        cls.fix_who = None

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckKDRA, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        inp, oup = inplanes, planes * 4
        self.squeeze = inp // 16
        self.dim = int(math.sqrt(inp))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp, self.squeeze, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.squeeze, 4, bias=False),
        )
        self.sf = nn.Softmax(dim=1)
        self.conv_s1 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s2 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s3 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s4 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)

    def shuffle_col(self, data):
        cdata = data.clone()
        B = data.shape[0]
        C = data.shape[1]
        for i in range(B):
            ridxs = torch.randperm(C)
            # print(ridxs)
            cdata[i] = data[i][ridxs]
        return cdata

    def forward(self, x):
        b, c, h, w = x.size()
        if self.fix_who == 'meta':
            y = self.fc(self.avg_pool(x).view(b, c)).view(b, 4, 1, 1, 1)
            y = self.sf(y)
            if share_dict['args'].pe_type == 'CI':
                # CI-PE
                rand_idxs = torch.randperm(y.size(0))
                y = y[rand_idxs]
            elif share_dict['args'].pe_type == 'CK':
                # CK-PE
                y = self.shuffle_col(y)
            else:
                raise ValueError(f'unknown pe type:{share_dict["args"].pe_type}')
        else:
            y = self.fc(self.avg_pool(x).view(b, c)).view(b, 4, 1, 1, 1)
            y = self.sf(y)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        dyres = self.conv_s1(out) * y[:, 0] + self.conv_s2(out) * y[:, 1] + \
                self.conv_s3(out) * y[:, 2] + self.conv_s4(out) * y[:, 3]
        out = dyres + self.conv2(out)
        # out = dyres + self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # + dyres
        out = self.relu(out)

        return out


def get_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    init_params = model.init_param
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in init_params:
                yield param


def get_10xparams(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    new_params = model.new_param
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in new_params:
                yield param


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)



"""
Residual networks with draac_v4 and mixstyle
"""


@BACKBONE_REGISTRY.register()
def resnet18_draac_v4_ms_l123(pretrain=None, **kwargs):
    from dassl.modeling.ops import MixStyle

    net = ResNetDRA(
        block=BasicBlockKDRN,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3'],
        pretrain=pretrain
    )

    return net


@BACKBONE_REGISTRY.register()
def resnet18_draac_v4_ms_l12(pretrain=None, **kwargs):
    from dassl.modeling.ops import MixStyle

    net = ResNetDRA(
        block=BasicBlockKDRN,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2'],
        pretrain=pretrain
    )

    return net


@BACKBONE_REGISTRY.register()
def resnet18_draac_v4_ms_l1(pretrain=None, **kwargs):
    from dassl.modeling.ops import MixStyle

    net = ResNetDRA(
        block=BasicBlockKDRN,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=['layer1'],
        pretrain=pretrain
    )

    return net


@BACKBONE_REGISTRY.register()
def resnet18_draac_v4(pretrain=None, **kwargs):
    net = ResNetDRA(
        block=BasicBlockKDRN,
        layers=[2, 2, 2, 2],
        pretrain=pretrain
    )

    return net


@BACKBONE_REGISTRY.register()
def resnet50_draac_v4_ms_l3(pretrain=None, **kwargs):
    from dassl.modeling.ops import MixStyle

    net = ResNetDRA(
        block=BottleneckKDRA,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer3'],
        pretrain=pretrain
    )

    return net


@BACKBONE_REGISTRY.register()
def resnet50_draac_v4_ms_l123(pretrain=None, **kwargs):
    from dassl.modeling.ops import MixStyle

    net = ResNetDRA(
        block=BottleneckKDRA,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3'],
        pretrain=pretrain
    )

    return net


@BACKBONE_REGISTRY.register()
def resnet50_draac_v4_ms_l12(pretrain=None, **kwargs):
    from dassl.modeling.ops import MixStyle

    net = ResNetDRA(
        block=BottleneckKDRA,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2'],
        pretrain=pretrain
    )

    return net


@BACKBONE_REGISTRY.register()
def resnet50_draac_v4_ms_l1(pretrain=None, **kwargs):
    from dassl.modeling.ops import MixStyle

    net = ResNetDRA(
        block=BottleneckKDRA,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=['layer1'],
        pretrain=pretrain
    )

    return net


@BACKBONE_REGISTRY.register()
def resnet50_draac_v4(pretrain=None, **kwargs):
    net = ResNetDRA(
        block=BottleneckKDRA,
        layers=[3, 4, 6, 3],
        pretrain=pretrain
    )

    return net


@BACKBONE_REGISTRY.register()
def resnet101_draac_v4_ms_l123(pretrain=None, **kwargs):
    from dassl.modeling.ops import MixStyle

    net = ResNetDRA(
        block=BottleneckKDRA,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2', 'layer3'],
        pretrain=pretrain
    )

    return net


@BACKBONE_REGISTRY.register()
def resnet101_draac_v4_ms_l12(pretrain=None, **kwargs):
    from dassl.modeling.ops import MixStyle

    net = ResNetDRA(
        block=BottleneckKDRA,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=['layer1', 'layer2'],
        pretrain=pretrain
    )

    return net


@BACKBONE_REGISTRY.register()
def resnet101_draac_v4_ms_l1(pretrain=None, **kwargs):
    from dassl.modeling.ops import MixStyle

    net = ResNetDRA(
        block=BottleneckKDRA,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=['layer1'],
        pretrain=pretrain
    )

    return net


@BACKBONE_REGISTRY.register()
def resnet101_draac_v4(pretrain=None, **kwargs):
    net = ResNetDRA(
        block=BottleneckKDRA,
        layers=[3, 4, 23, 3],
        pretrain=pretrain
    )

    return net
