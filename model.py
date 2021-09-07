import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet50


class Model(nn.Module):
    def __init__(self, backbone_type, proj_dim):
        super(Model, self).__init__()

        # backbone
        backbone = resnet50(pretrained=True) if backbone_type == 'resnet50' else vgg16(pretrained=True)
        extractor = []
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc', 'classifier']:
                extractor.append(module)
        self.extractor = nn.Sequential(*extractor)

        # proj
        self.proj = nn.Linear(2048 if backbone_type == 'resnet50' else 512, proj_dim)

    def forward(self, img):
        feat = self.extractor(img)

        feat = torch.flatten(F.adaptive_max_pool2d(feat, (1, 1)), start_dim=1)
        proj = self.proj(feat)

        return F.normalize(proj, dim=-1)
