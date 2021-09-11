import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet50


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Encoder, self).__init__()

        hid_dim = int((in_dim + out_dim) / 2)

        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        out = self.fc(x)
        return F.normalize(out, dim=-1)


class Generator(nn.Module):
    def __init__(self, in_dim):
        super(Generator, self).__init__()

        self.fc = nn.Linear(in_dim, 512 * 4 * 4)

        self.convT_1 = nn.ConvTranspose2d(512, 256, 5, 2, padding=2)
        self.bn_1 = nn.InstanceNorm2d(256)
        self.convT_2 = nn.ConvTranspose2d(256, 128, 5, 2, padding=2, output_padding=1)
        self.bn_2 = nn.InstanceNorm2d(128)
        self.convT_3 = nn.ConvTranspose2d(128, 64, 5, 2, padding=2, output_padding=1)
        self.bn_3 = nn.InstanceNorm2d(64)
        self.convT_4 = nn.ConvTranspose2d(64, 32, 5, 2, padding=2, output_padding=1)
        self.bn_4 = nn.InstanceNorm2d(32)
        self.convT_5 = nn.ConvTranspose2d(32, 16, 5, 2, padding=2, output_padding=1)
        self.bn_5 = nn.InstanceNorm2d(16)
        self.convT_6 = nn.ConvTranspose2d(16, 8, 5, 2, padding=2, output_padding=1)
        self.bn_6 = nn.InstanceNorm2d(8)
        self.convT_7 = nn.ConvTranspose2d(8, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(-1, 512, 4, 4)

        x = self.relu(self.bn_1(self.convT_1(x)))
        x = self.relu(self.bn_2(self.convT_2(x)))
        x = self.relu(self.bn_3(self.convT_3(x)))
        x = self.relu(self.bn_4(self.convT_4(x)))
        x = self.relu(self.bn_5(self.convT_5(x)))
        x = self.relu(self.bn_6(self.convT_6(x)))
        out = self.tanh(self.convT_7(x))

        return out


class Extractor(nn.Module):
    def __init__(self, backbone_type):
        super(Extractor, self).__init__()

        # backbone
        backbone = resnet50(pretrained=True) if backbone_type == 'resnet50' else vgg16(pretrained=True)
        extractor = []
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc', 'classifier']:
                extractor.append(module)
        self.backbone = nn.Sequential(*extractor)

    def forward(self, img):
        feat = self.backbone(img)
        feat = torch.flatten(F.adaptive_max_pool2d(feat, (1, 1)), start_dim=1)
        return F.normalize(feat, dim=-1)
