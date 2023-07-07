import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, block_in, branch_out):
        super(InceptionBlock, self).__init__()
        self.branch_x1 = nn.Sequential(nn.Conv2d(block_in, branch_out[0], kernel_size=1),
                                       nn.ReLU())
        self.branch_x3 = nn.Sequential(nn.Conv2d(block_in, branch_out[1][0], kernel_size=1),
                                       nn.ReLU(),
                                       nn.Conv2d(branch_out[1][0], branch_out[1][1], kernel_size=3, padding=1),
                                       nn.ReLU())
        self.branch_x5 = nn.Sequential(nn.Conv2d(block_in, branch_out[2][0], kernel_size=1),
                                       nn.ReLU(),
                                       nn.Conv2d(branch_out[2][0], branch_out[2][1], kernel_size=5, padding=2),
                                       nn.ReLU())
        self.branch_proj = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                         nn.Conv2d(block_in, branch_out[3], kernel_size=1),
                                         nn.ReLU())

    def forward(self, x):
        br_out0 = self.branch_x1(x)
        br_out1 = self.branch_x3(x)
        br_out2 = self.branch_x5(x)
        br_out3 = self.branch_proj(x)
        outputs = [br_out0, br_out1, br_out2, br_out3]
        return torch.cat(outputs, dim=1)


'''
GoogleNet:
feature map cal method: (W0 + 2 x padding - kernel_size)/ stride + 1
1. Input:3 x 224 x 224
2. Stage1:
    conv(7, 2, 3) - 64
    relu()
    maxpool(3x3, 2) - 64
    112 x 112
3. Stage2:
    conv(1): - 64
    conv(3) - 192
    relu()
    maxpool(3x3, 2) - 192
    56 x 56
4. Sa
'''
class GoogleNet(nn.Module):
    """
    Backbone of Googlenet
    The GoogleNet is the one in 2015CVPR Going deeper with convolutions.
    Notice this model does not contain task specific layers.
    """
    def __init__(self, with_aux=False):
        super(GoogleNet, self).__init__()
        self.with_aux = with_aux
        self.lrn = nn.LocalResponseNorm(5)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                   nn.ReLU())
        self.incp3a = InceptionBlock(192, [64, (96, 128), (16, 32), 32])
        self.incp3b = InceptionBlock(256, [128, (128, 192), (32, 96), 64])
        self.incp4a = InceptionBlock(480, [192, (96, 208), (16, 48), 64])
        self.incp4b = InceptionBlock(512, [160, (112, 224), (24, 64), 64])
        self.incp4c = InceptionBlock(512, [128, (128, 256), (24, 64), 64])
        self.incp4d = InceptionBlock(512, [112, (144, 288), (32, 64), 64])
        self.incp4e = InceptionBlock(528, [256, (160, 320), (32, 128), 128])
        self.incp5a = InceptionBlock(832, [256, (160, 320), (32, 128), 128])
        self.incp5b = InceptionBlock(832, [384, (192, 384), (48, 128), 128])

    def get_layer(self):
        layers = []
        layers.append(self.conv1)
        layers.append(self.max_pool)
        layers.append(self.lrn)
        layers.append(self.conv2)
        layers.append(self.lrn)
        layers.append(self.max_pool)
        layers.append(self.incp3a)
        layers.append(self.incp3b)
        layers.append(self.max_pool)
        layers.append(self.incp4a)
        layers.append(self.incp4b)
        layers.append(self.incp4c)
        layers.append(self.incp4d)
        layers.append(self.incp4e)
        layers.append(self.incp5a)
        layers.append(self.incp5b)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x) # 112 x 112 x 64
        x = self.max_pool(x) # 56 x 56 x 64
        x = self.lrn(x)
        x = self.conv2(x) # 56 x 56 x 192
        x = self.lrn(x)
        x = self.max_pool(x) # 28 x 28 x 192
        x = self.incp3a(x) # 28 x 28 x 256
        x = self.incp3b(x) # 28 x 28 x 480
        x = self.max_pool(x) # 14 x 14 x 480
        x = self.incp4a(x) # 14 x 14 x 512
        aux1 = x
        x = self.incp4b(x) #
        x = self.incp4c(x)
        x = self.incp4d(x)
        aux2 = x
        x = self.incp4e(x)
        x = self.max_pool(x)
        x = self.incp5a(x)
        x = self.incp5b(x)
        aux3 = x
        if self.training and self.with_aux:
            return aux1, aux2, aux3
        return aux3