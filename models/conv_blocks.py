import torch.nn as nn
import torch


class ResidualAdapter(nn.Module):
    def __init__(self, block, stride=1, relu_after_shortcut=False, use_1x1=False, in_dim=-1, out_dim=-1, no_pool_downsampling=False):
        super(ResidualAdapter, self).__init__()

        if use_1x1:
            self.conv_for_res = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.conv_for_res = None

        if stride > 1:
            if no_pool_downsampling:
                # this is used for the ResNet Basic Block
                self.downsample = SimpleSubsampling()
            else:
                self.downsample = nn.AvgPool2d(
                    (2, 2), stride=(2, 2), ceil_mode=True)
        else:
            self.downsample = None
        self.block = block

        self.relu = nn.ReLU()
        self.relu_after_shortcut = relu_after_shortcut
        self.add = Add()

    def forward(self, x):
        out = self.block(x)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        if self.conv_for_res is not None:
            # in case residual_channel < shortcut channel
            shortcut = self.conv_for_res(shortcut)

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.zeros(
                batch_size,
                residual_channel - shortcut_channel,
                featuremap_size[0],
                featuremap_size[1]
            )
            if out.is_cuda:
                padding = padding.cuda()

            out = self.add(out, torch.cat((shortcut, padding), 1))
        else:
            out = self.add(out, shortcut)

        # old ResNet Basic Block uses ReLU after shortcut
        if self.relu_after_shortcut:
            return self.relu(out)
        else:
            return out


class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride):
        super(BasicBlock, self).__init__()

        convolutions = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=k,
                      stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=k,
                      stride=1, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        self.block_with_shortcut = ResidualAdapter(
            convolutions, stride=stride, relu_after_shortcut=True,
            no_pool_downsampling=True
        )

    def forward(self, x):
        # see He Deep Residual Learning Figure 2
        return self.block_with_shortcut(x)


class PyramidBasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride):
        super(PyramidBasicBlock, self).__init__()

        convolutions = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=k,
                      stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=k,
                      stride=1, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        self.block_with_shortcut = ResidualAdapter(
            convolutions, stride=stride, relu_after_shortcut=False
        )

    def forward(self, x):
        return self.block_with_shortcut(x)


class FPBlockJOV(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q, use_1x1=False):
        super(FPBlockJOV, self).__init__()

        fp_block = _FPBlockJOV(in_dim, out_dim, k=k, stride=stride, q=q)
        self.block_with_shortcut = ResidualAdapter(
            fp_block, stride=stride,
            relu_after_shortcut=False, use_1x1=use_1x1,
            in_dim=in_dim, out_dim=out_dim
        )

    def forward(self, x):
        return self.block_with_shortcut(x)


class _FPBlockJOV(nn.Module):
    def __init__(self, in_dim, out_dim, *, k, stride, q):
        super(_FPBlockJOV, self).__init__()

        feat_dim = int(q * out_dim)
        self.upper = nn.Sequential(
            nn.Conv2d(in_dim, feat_dim, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU()
        )
        self.lower = nn.Sequential(
            nn.Conv2d(feat_dim, out_dim, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

        self.left_dw = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=k, stride=stride,
                      padding=k // 2, bias=False, groups=feat_dim),
            nn.InstanceNorm2d(feat_dim),
            nn.ReLU()
        )
        self.right_dw = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=k, stride=stride,
                      padding=k // 2, bias=False, groups=feat_dim),
            nn.InstanceNorm2d(feat_dim),
            nn.ReLU()
        )

        self.mult = Multiplication()

    def forward(self, x):
        x = self.upper(x)
        x_left = self.left_dw(x)
        x_right = self.right_dw(x)

        x = self.mult(x_left, x_right)

        x = self.lower(x)

        return x


class MinBlock(FPBlockJOV):
    def __init__(self, in_dim, out_dim, *, k, stride, q, use_1x1=False):
        super(MinBlock, self).__init__(
            in_dim, out_dim, k=k, stride=stride, q=q, use_1x1=use_1x1
        )
        # change the multiplication
        self.block_with_shortcut.block.mult = Minimum()


class Multiplication(nn.Module):
    def forward(self, x, y):
        return x * y


class Add(nn.Module):
    """Simple add operation for
    better readability in TensorBoard
    """

    def forward(self, a, b):
        return a + b


class SimpleSubsampling(nn.Module):
    def forward(self, x):
        return x[..., ::2, ::2]


class Minimum(nn.Module):
    def forward(self, x, y):
        return torch.minimum(x, y)
