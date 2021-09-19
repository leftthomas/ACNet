import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet50


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1), nn.Conv2d(in_channels, in_channels, 3), nn.InstanceNorm2d(in_channels),
                      nn.ReLU(inplace=True), nn.ReflectionPad2d(1), nn.Conv2d(in_channels, in_channels, 3),
                      nn.InstanceNorm2d(in_channels)]

        self.conv = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv(x)


class Generator(nn.Module):
    def __init__(self, in_channels=32, num_block=9):
        super(Generator, self).__init__()

        # in conv
        self.in_conv = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, 32, 7), nn.InstanceNorm2d(32),
                                     nn.ReLU(inplace=True))

        # down sample
        down_sample = []
        for _ in range(2):
            out_channels = in_channels * 2
            down_sample += [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                            nn.InstanceNorm2d(out_channels), nn.ReLU(inplace=True)]
            in_channels = out_channels
        self.down_sample = nn.Sequential(*down_sample)

        # conv blocks
        self.convs = nn.Sequential(*[ResidualBlock(in_channels) for _ in range(num_block)])

        # up sample
        up_sample = []
        for _ in range(2):
            out_channels = in_channels // 2
            up_sample += [nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                          nn.InstanceNorm2d(out_channels), nn.ReLU(inplace=True)]
            in_channels = out_channels
        self.up_sample = nn.Sequential(*up_sample)

        # out conv
        self.out_conv = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(32, 3, 7), nn.Tanh())

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down_sample(x)
        x = self.convs(x)
        x = self.up_sample(x)
        out = self.out_conv(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=32):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, in_channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels * 2, 4, stride=2, padding=1),
                                   nn.InstanceNorm2d(in_channels * 2), nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels * 4, 4, stride=2, padding=1),
                                   nn.InstanceNorm2d(in_channels * 4), nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, in_channels * 8, 4, padding=1),
                                   nn.InstanceNorm2d(in_channels * 8), nn.LeakyReLU(0.2, inplace=True))

        self.conv5 = nn.Conv2d(in_channels * 8, 1, 4, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.conv5(x)
        return out


class Extractor(nn.Module):
    def __init__(self, backbone_type, emb_dim):
        super(Extractor, self).__init__()

        # backbone
        backbone = resnet50(pretrained=True) if backbone_type == 'resnet50' else vgg16(pretrained=True)
        extractor = []
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc', 'classifier']:
                extractor.append(module)
        self.backbone = nn.Sequential(*extractor)
        self.fc = nn.Linear(2048 if backbone_type == 'resnet50' else 512, emb_dim)

    def forward(self, img):
        feat = self.backbone(img)
        feat = torch.flatten(F.adaptive_max_pool2d(feat, (1, 1)), start_dim=1)
        out = self.fc(feat)
        return F.normalize(out, dim=-1)
