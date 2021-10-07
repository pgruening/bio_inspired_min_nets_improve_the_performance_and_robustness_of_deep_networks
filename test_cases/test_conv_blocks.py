import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

sys.path.append('..')


class TestResNetBasicBlock(unittest.TestCase):

    def test_same_block_output(self):
        import models.conv_blocks

        # this does not work with stride 2 because the shortcut is different:
        # res adapter uses avgpool reduction
        # Ydelbayev basic block uses sub-sampling (y[...,::2, ::2])
        original_block = BasicBlock(16, 32, stride=2, option='A').eval()
        new_block = models.conv_blocks.BasicBlock(
            16, 32, k=3, stride=2
        ).eval()

        with torch.no_grad():
            # set equal weights for both blocks
            wc1 = original_block.conv1.weight
            new_block.block_with_shortcut.block[0].weight = wc1
            assert new_block.block_with_shortcut.block[0].bias is None

            wc2 = original_block.conv2.weight
            new_block.block_with_shortcut.block[3].weight = wc2
            assert new_block.block_with_shortcut.block[3].bias is None

            for _ in range(10):
                x = torch.rand(1, 16, 16, 16)

                self.assertEqual(
                    torch.abs(original_block(x) - new_block(x)).sum().item(),
                    0.
                )


class TestPyramidNetBasicBlock(unittest.TestCase):
    def test_same_block_output(self):
        import models.conv_blocks

        stride = 2
        # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
        if stride != 1:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)
        original_block = PyramidBasicBlock(
            16, 32, stride=stride, downsample=downsample
        ).eval()
        new_block = models.conv_blocks.PyramidBasicBlock(
            16, 32, k=3, stride=2
        ).eval()

        with torch.no_grad():
            # set equal weights for both blocks
            wc1 = original_block.conv1.weight
            new_block.block_with_shortcut.block[1].weight = wc1
            assert new_block.block_with_shortcut.block[1].bias is None

            wc2 = original_block.conv2.weight
            new_block.block_with_shortcut.block[4].weight = wc2
            assert new_block.block_with_shortcut.block[4].bias is None

            for _ in range(10):
                x = torch.rand(1, 16, 16, 16)

                self.assertEqual(
                    torch.abs(original_block(x) - new_block(x)).sum().item(),
                    0.
                )


# https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # this short
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 0, planes // 2), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/PyramidNet.py


class PyramidBasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PyramidBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.FloatTensor(
                batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


if __name__ == '__main__':
    unittest.main()
