import random

import numpy as np
import torch
from PIL import Image
from torch.backends import cudnn
from torchvision.transforms import ToPILImage

from model import Generator
from utils import get_transform

# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False

if __name__ == '__main__':
    generator = Generator(in_channels=8, num_block=8).cuda()
    generator.load_state_dict(torch.load('result/sketchy_resnet50_512_generator.pth'))
    generator.eval()
    img = Image.open('result/test.png').convert('RGB')
    syns_img = generator(get_transform('val')(img).unsqueeze(dim=0).cuda())
    syns_img = torch.clamp(((syns_img + 1) / 2) * 255, min=0, max=255).byte()
    syns_img = ToPILImage()(syns_img.squeeze(dim=0).cpu())
    syns_img.save('result/syns.jpg')
