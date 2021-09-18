import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet50


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, n_residual_blocks=9):
        super(Generator, self).__init__()

        # initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, 32, 7),
                 nn.InstanceNorm2d(32),
                 nn.ReLU(inplace=True)]

        # down sampling
        in_features = 32
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # up sampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(32, 3, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # a bunch of convolutions one after another
        model = [nn.Conv2d(3, 32, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(32, 64, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(256, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


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
