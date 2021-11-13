import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils import get_transform


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


if __name__ == '__main__':
    sketch_paths = glob.glob('/home/rh/Downloads/sketchy/train/sketch/*/*_real.png')
    photo_paths = glob.glob('/home/rh/Downloads/sketchy/train/photo/*/*_real.png')
    model = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='max').cuda()
    model.eval()
    transform = get_transform(split='test')

    ref_emb = np.array([1.0, 0.0])
    sketches, photos = [], []
    with torch.no_grad():
        for sketch_path in tqdm(sketch_paths, desc='processing sketches'):
            sketch = transform(Image.open(sketch_path)).unsqueeze(dim=0)
            sketch_emd = F.normalize(model(sketch.cuda()), dim=-1).squeeze(dim=0).cpu()
            sketches.append(sketch_emd)
        for photo_path in tqdm(photo_paths, desc='processing photos'):
            photo = transform(Image.open(photo_path)).unsqueeze(dim=0)
            photo_emd = F.normalize(model(photo.cuda()), dim=-1).squeeze(dim=0).cpu()
            photos.append(photo_emd)
        sketches = torch.stack(sketches, dim=0).numpy()
        photos = torch.stack(photos, dim=0).numpy()

        tsne = TSNE(n_components=2, init='pca', random_state=0)
        sketches = tsne.fit_transform(sketches)
        photos = tsne.fit_transform(photos)

        sketch_angs, photo_angs = [], []
        for sketch in sketches:
            sketch_ang = angle_between(sketch, ref_emb)
            sketch_angs.append(sketch_ang)
        for photo in photos:
            photo_ang = angle_between(photo, ref_emb)
            photo_angs.append(photo_ang)

        sns.kdeplot(np.array(sketch_angs), shade=True, label='sketch dist.')
        sns.kdeplot(np.array(photo_angs), shade=True, label='photo dist.')
        plt.legend()
        plt.savefig('result/before.pdf')
