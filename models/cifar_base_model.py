import warnings

import torch.nn as nn
import torch.nn.init as init
import torch

from . import conv_blocks
import math


def get_model(model_type, in_dim, out_dim, device, **model_kwargs):
    if model_type == 'CifarResNet':
        N = int(model_kwargs['N'][0])
        assert N > 0 and N < 20
        get_block = get_block_adapter('BasicBlock')
        model = BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N, init_normal=True
        )

    elif model_type == 'CifarPyrResNet':
        N = int(model_kwargs['N'][0])
        assert N > 0 and N < 20
        get_block = get_block_adapter('PyramidBasicBlock')
        model = BaseModel(
            in_dim, out_dim, block_getter=get_block, N=N, pyr_weight_init=True
        )

    elif model_type == 'CifarMinFP-LS':
        N = int(model_kwargs['N'][0])
        assert N > 0 and N < 20

        q = model_kwargs['q'][0]
        q = float(q)

        default_block = get_block_adapter('PyramidBasicBlock')
        start_block = get_block_adapter('MinBlock', q=q)

        model = JOVFPAtStackStart(
            in_dim, out_dim, default_block=default_block,
            start_block=start_block, N=N
        )

    elif model_type == 'CifarMinFP-LS-RNBasic':
        N = int(model_kwargs['N'][0])
        assert N > 0 and N < 20

        q = model_kwargs['q'][0]
        q = float(q)

        default_block = get_block_adapter('BasicBlock')
        start_block = get_block_adapter('MinBlock', q=q)

        model = JOVFPAtStackStart(
            in_dim, out_dim, default_block=default_block,
            start_block=start_block, N=N
        )

    else:
        raise ValueError(f'Unknown Model: {model_type}')

    return model.to(device).eval()


def get_block_adapter(block_type, **kwargs):
    # for most block the bias is automatically set to false
    if block_type == 'BasicBlock':
        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return conv_blocks.BasicBlock(
                in_dim, out_dim, k=k, stride=stride
            )
    elif block_type == 'PyramidBasicBlock':
        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return conv_blocks.PyramidBasicBlock(
                in_dim, out_dim, k=k, stride=stride
            )

    elif block_type == 'MinBlock':
        q = kwargs["q"]
        assert isinstance(q, (float, int))

        def get_block(in_dim, out_dim, *, k, stride, padding, bias):
            return conv_blocks.MinBlock(
                in_dim, out_dim, k=k, stride=stride, q=q
            )
    else:
        raise ValueError(f'Unknown block: {block_type}')

    return get_block


class BaseModel(nn.Module):
    def __init__(self, in_dim, out_dim, *, block_getter, N, init_normal=False, pyr_weight_init=False, pre_transform=None):
        super(BaseModel, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_dim, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # create three stacks, all with the same number and type of blocks
        # each first block of a stack may change the number of features and
        # the stride.

        self.stack1 = nn.Sequential()
        for i in range(N):
            self.stack1.add_module(
                f'block_{i}',
                block_getter(16, 16, k=3, stride=1, padding=1, bias=False),
            )

        self.stack2 = nn.Sequential()
        self.stack2.add_module(
            f'block_{0}',
            block_getter(16, 32, k=3, stride=2, padding=1, bias=False),
        )
        for i in range(1, N):
            self.stack2.add_module(
                f'block_{i}',
                block_getter(32, 32, k=3, stride=1, padding=1, bias=False),
            )

        self.stack3 = nn.Sequential()
        self.stack3.add_module(
            f'block_{0}',
            block_getter(32, 64, k=3, stride=2, padding=1, bias=False),
        )
        for i in range(1, N):
            self.stack3.add_module(
                f'block_{i}',
                block_getter(64, 64, k=3, stride=1, padding=1, bias=False),
            )

        self.out = nn.Linear(64, out_dim, bias=True)

        if init_normal:
            warnings.warn('Overwriting weight init with Kaminig-Normal')
            self.apply(_weights_init)

        if pyr_weight_init:
            warnings.warn(
                'Overwriting weight init with Kaminig-Normal Fan-out(PyramidNet)'
            )
            _pyr_weight_init(self)

        self.pre_transform = pre_transform

    def forward(self, x):
        if self.pre_transform is not None:
            # torchvision 0.7: no transforms over entire batch
            x = [self.pre_transform(x[i, ...]) for i in range(x.shape[0])]
            x = torch.stack(x, 0)

        x = self.first_conv(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        # global average pooling
        z = x.mean([2, 3])
        y = self.out(z)
        return y


class JOVFPAtStackStart(BaseModel):
    def __init__(self, in_dim, out_dim, *, default_block, start_block, N):
        super(JOVFPAtStackStart, self).__init__(
            in_dim, out_dim, block_getter=default_block, N=N)

        # change each first block of a stack
        self.stack1[0] = start_block(
            16, 16, k=3, stride=1, padding=1, bias=False)
        self.stack2[0] = start_block(
            16, 32, k=3, stride=2, padding=1, bias=False)
        self.stack3[0] = start_block(
            32, 64, k=3, stride=2, padding=1, bias=False)

# -----------------------------------------------------------------------------
# ---------------------------WEIGHT INIT---------------------------------------
# -----------------------------------------------------------------------------

# in the original He ResNet paper the initialization was He-Normal


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def _pyr_weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
