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
