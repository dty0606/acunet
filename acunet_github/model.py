import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from collections import OrderedDict

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.hsigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.hsigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(), #default r i s 4
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)



class ACUNet(nn.Module):

    def __init__(self, in_channels=2, out_channels=1, init_features=32, dropout=0.5):
        super(ACUNet, self).__init__()

        block = InvertedResidual

        '''ACU-Net'''

        self.encoder1 = conv_3x3_bn(2, 16, 2) #256, #layer1
        self.encoder2 = nn.Sequential(
            block(16, 1*16, 16, 3, 1, 0, 0), #128, #layer2
        ) 
        self.encoder3 = nn.Sequential(
            block(16, 4*16, 24, 3, 2, 0, 0), #64, #layer3
            block(24, 3*24, 24, 3, 1, 0, 0), #64, #layer3
            )
        self.encoder4 = nn.Sequential(
            block(24, 4*24, 40, 5, 2, 1, 0), #32, #layer3
            block(40, 6*40, 40, 5, 1, 1, 1), #32, #layer4
            block(40, 6*40, 40, 5, 1, 1, 1), #32, #layer4
            )

        self.bottleneck = block(40, 6*40, 240, 5, 2, 1, 1) #16, #layer4

        #unsymetric right side
        self.upconv4 = nn.ConvTranspose2d(240, 40, kernel_size=2, stride=2)#32, #layer1
        self.decoder4 = block(80, 6*40, 40, 5, 1, 1, 1) #32, #layer3
        self.upconv3 = nn.ConvTranspose2d(40, 24, kernel_size=2, stride=2)#layer1
        self.decoder3 = block(48, 3*24, 24, 5, 1, 1, 1) #64 #layer3
        self.upconv2 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)#layer1
        self.decoder2 = block(32, 1*16, 16, 3, 1, 1, 1) #128 #layer3
        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)#layer1
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        # self.drop = nn.Dropout(p=dropout)


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2) #256*1
        dec1 = self.conv(dec1)
        fc = torch.sigmoid(dec1)
        # x = self.drop(fc)

        return fc


class ACUNet_md(nn.Module):

    def __init__(self, in_channels=2, out_channels=1, init_features=32, dropout=0.5):
        super(ACUNet_md, self).__init__()

        block = InvertedResidual

        self.encoder1 = conv_3x3_bn(2, 64, 2) #256, #layer1
        self.encoder2 = nn.Sequential(
            block(64, 4*64, 64, 3, 1, 0, 0), #128, #layer2
        ) 
        self.encoder3 = nn.Sequential(
            block(64, 3*128, 128, 3, 2, 0, 0), #64, #layer3
            block(128, 4*128, 128, 3, 1, 0, 0), #64, #layer3
            )
        self.encoder4 = nn.Sequential(
            block(128, 4*128, 256, 5, 2, 1, 0), #32, #layer3
            block(256, 6*256, 256, 5, 1, 1, 1), #32, #layer4
            block(256, 6*256, 256, 5, 1, 1, 1), #32, #layer4
            )

        self.bottleneck = block(256, 6*256, 512, 5, 2, 1, 1) #16, #layer4

        #unsymetric right side
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)#32, #layer1
        self.decoder4 = block(512, 6*256, 256, 5, 1, 1, 1) #32, #layer3
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)#layer1
        self.decoder3 = block(256, 3*128, 128, 5, 1, 1, 1) #64 #layer3
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)#layer1
        self.decoder2 = block(128, 3*64, 64, 3, 1, 1, 1) #128 #layer3
        self.upconv1 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)#layer1
        # self.conv = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1)
        # self.drop = nn.Dropout(p=dropout)
        # self.drop = nn.Dropout(p=dropout)
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
       


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2) #256*1
        dec1 = self.conv(dec1)
        fc = torch.sigmoid(dec1)
        # x = self.drop(fc)

        return fc


class ACUNet_lg(nn.Module):

    def __init__(self, in_channels=2, out_channels=1, init_features=32, dropout=0.5):
        super(ACUNet_lg, self).__init__()

        block = InvertedResidual

        self.encoder1 = conv_3x3_bn(2, 64, 2) #256, #layer1
        self.encoder2 = nn.Sequential(
            block(64, 4*64, 64, 3, 1, 0, 0), #128, #layer2
        ) 
        self.encoder3 = nn.Sequential(
            block(64, 3*128, 128, 3, 2, 0, 0), #64, #layer3
            block(128, 4*128, 128, 3, 1, 0, 0), #64, #layer3
            )
        self.encoder4 = nn.Sequential(
            block(128, 4*128, 256, 5, 2, 1, 0), #32, #layer3
            block(256, 6*256, 256, 5, 1, 1, 1), #32, #layer4
            block(256, 6*256, 256, 5, 1, 1, 1), #32, #layer4
            )

        self.encoder5 = nn.Sequential(
            block(256, 4*256, 512, 5, 2, 1, 0), #32, #layer3
            block(512, 6*256, 512, 5, 1, 1, 1), #32, #layer4
            block(512, 6*256, 512, 5, 1, 1, 1), #32, #layer4
            )

        self.bottleneck = block(512, 4*512, 1024, 5, 2, 1, 1) #16, #layer4

        #unsymetric right side
        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)#32, #layer1
        self.decoder5 = block(1024, 4*512, 512, 5, 1, 1, 1) #32, #layer3
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)#layer1
        self.decoder4 = block(512, 6*256, 256, 5, 1, 1, 1) #64 #layer3
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)#layer1
        self.decoder3 = block(256, 4*128, 128, 3, 1, 1, 1) #128 #layer3
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)#layer1
        self.decoder2 = block(128, 3*64, 64, 3, 1, 1, 1) #128 #layer3
        self.upconv1 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)#layer1
        # self.drop = nn.Dropout(p=dropout)
        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
       


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        bottleneck = self.bottleneck(enc5)

        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2) #256*1
        dec1 = self.conv(dec1)
        fc = torch.sigmoid(dec1)
        # x = self.drop(fc)

        return fc


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )