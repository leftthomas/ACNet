import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils import get_transform

if __name__ == '__main__':
    data_name = 'sketchy'
    data_root = '/home/rh/Downloads/{}/train'.format(data_name)

    old_paths, new_paths, photo_paths = [], [], []
    for class_name in sorted(glob.glob('{}/sketch/*'.format(data_root)))[:50]:
        class_name = class_name.split('/')[-1]
        old_paths += sorted(glob.glob('{}/sketch/{}/*_real.png'.format(data_root, class_name)))[:10]
        new_paths += sorted(glob.glob('{}/sketch/{}/*_fake.png'.format(data_root, class_name)))[:10]
        photo_paths += sorted(glob.glob('{}/photo/{}/*_real.png'.format(data_root, class_name)))[:10]
    model = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='max').cuda()
    model.eval()
    transform = get_transform(split='test')

    old_sketches, new_sketches, photos = [], [], []
    for paths, embeds in zip([old_paths, new_paths, photo_paths], [old_sketches, new_sketches, photos]):
        for path in tqdm(paths, desc='processing data'):
            emd = transform(Image.open(path)).unsqueeze(dim=0)
            with torch.no_grad():
                embeds.append(model(emd.cuda()).squeeze(dim=0).cpu())

    old_sketches = torch.stack(old_sketches)
    new_sketches = torch.stack(new_sketches)
    photos = torch.stack(photos)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    old_sketches = tsne.fit_transform(old_sketches.numpy())
    new_sketches = tsne.fit_transform(new_sketches.numpy())
    photos = tsne.fit_transform(photos.numpy())

    embeds = np.concatenate((old_sketches, new_sketches, photos))
    x_min, x_max = np.min(embeds, 0), np.max(embeds, 0)
    embeds = (embeds - x_min) / (x_max - x_min)
    labels = ['old'] * len(old_sketches) + ['new'] * len(new_sketches) + ['photo'] * len(photos)
    data = pd.DataFrame({'x': embeds[:, 0].tolist(), 'y': embeds[:, 1].tolist(), 'domain': labels})

    sns.scatterplot(x='x', y='y', hue='domain', palette='Set2', alpha=0.4, data=data)
    plt.show()
    plt.savefig('result/{}_dist.pdf'.format(data_name), bbox_inches='tight', pad_inches=0.1)
