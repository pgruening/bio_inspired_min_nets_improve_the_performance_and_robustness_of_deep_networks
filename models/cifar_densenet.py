'''DenseNet in PyTorch.
    Adopted from https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py
    This repo is listed in the git repo from the original DenseNet(Lua).
    original paper: https://arxiv.org/pdf/1608.06993.pdf
'''

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

import models.conv_blocks as cb


def get_blocks(net_depth, is_bottleneck):
    # referring to https://github.com/liuzhuang13/DenseNetCaffe/issues/9
    # probably used for imagenet
    if net_depth == '121':
        return [6, 12, 24, 16]
    elif net_depth == '169':
        return [6, 12, 32, 32]
    elif net_depth == '201':
        return [6, 12, 48, 32]
    elif net_depth == '161':
        return [6, 12, 36, 24]
    else:
        return compute_cifar_blocks(net_depth, is_bottleneck)

# TODO Rename this here and in `get_blocks`


def compute_cifar_blocks(net_depth, is_bottleneck):
    # cifar models: L = depth = 3n+4
    l = int(net_depth)
    n_layers = (l - 4)

    if n_layers % 3 != 0:
        raise ValueError(f'Unknown Network Depth: {net_depth}')
    n = int(n_layers / 3)
    # from the original repo:
    # if bottleneck then N = N/2 end
    if is_bottleneck:
        n //= 2
    else:
        raise NotImplementedError
    return [n] * 3


def get_model(model_type, in_dim, out_dim, device, **model_kwargs):
    # imagenet_depths = ['121', '169', '201', '161']

    is_bottleneck = model_kwargs.get('is_bottleneck', [True])[0]
    n_blocks = get_blocks(model_kwargs['n_blocks'][0], is_bottleneck)

    growth_rate = model_kwargs['growth_rate'][0]
    growth_rate = int(growth_rate)

    # default arguments from https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py
    bn_size = int(model_kwargs.get('bn_size', [4])[0])
    drop_rate = int(model_kwargs.get('k', [0])[0])
    efficient = model_kwargs.get('efficient', [False])[0]

    # from repo:
    # channels before entering the first Dense-Block
    # local nChannels = 2 * growthRate

    num_init_features = 2 * growth_rate
    compression = float(model_kwargs.get('compression', [0.5])[0])

    if model_type == 'CifarDenseMin':
        start_block = get_block_adapter('MinBlock', **model_kwargs)
        default_block = get_block_adapter('DenseLayer')

    elif model_type == 'CifarDenseNet':
        start_block = get_block_adapter('DenseLayer')
        default_block = get_block_adapter('DenseLayer')
    else:
        raise ValueError(f'Unknown Model: {model_type}')

    model = DenseNet(
        start_block, default_block,
        input_dim=in_dim,
        num_classes=out_dim,
        growth_rate=growth_rate,
        block_config=n_blocks,
        compression=compression,
        num_init_features=num_init_features,
        bn_size=bn_size,
        drop_rate=drop_rate,
        small_inputs=True,
        efficient=efficient,
        is_bottleneck=is_bottleneck,
    )
    return model.to(device).eval()


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(self, start_block, default_block, input_dim=3, growth_rate=12, block_config=[16, 16, 16], compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0, is_bottleneck=True,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(input_dim, num_init_features,
                                    kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(input_dim, num_init_features,
                                    kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module(
                'norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                start_block, default_block,
                num_layers=num_layers,
                num_input_features=num_features, growth_rate=growth_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif (
                'norm' in name
                and 'bias' in name
                or 'classifier' in name
                and 'bias' in name
            ):
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def get_block_adapter(block_type, **kwargs):
    def fp_kwargs(in_dim, out_dim, kwargs):
        return {
            'q': float(kwargs['q'][0]),
            'k': int(kwargs.get('k', [3])[0]),
            'stride': int(kwargs.get('stride', [1])[0]),
            'use_1x1': in_dim > out_dim
        }
    if block_type == 'MinBlock':
        def get_block(in_dim, out_dim):
            return cb.MinBlock(
                in_dim, out_dim,
                **fp_kwargs(in_dim, out_dim, kwargs)
            )

    elif block_type == 'DenseLayer':
        bn_size = int(kwargs.get('bn_size', [4])[0])
        drop_rate = int(kwargs.get('k', [0])[0])
        efficient = kwargs.get('efficient', [False])

        def get_block(in_planes, growth_rate):
            return _DenseLayer(
                in_planes,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )

    else:
        raise(ValueError(f'Unknown block: {block_type}'))

    return get_block


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concat_features = torch.cat(inputs, 1)
        return conv(relu(norm(concat_features)))

    return bn_function


class _DenseBlock(nn.Module):
    def __init__(self, start_block, default_block, num_layers, num_input_features, growth_rate):
        super(_DenseBlock, self).__init__()

        layer = start_block(
            num_input_features,
            growth_rate
        )
        self.add_module('start_block', layer)

        for i in range(1, num_layers):
            layer = default_block(
                num_input_features + i * growth_rate,
                growth_rate
            )
            self.add_module('default_block%d' % i, layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)
