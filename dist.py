import argparse
import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Angle Dist.')
    parser.add_argument('--data_root', default='/data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='sketchy', type=str, choices=['sketchy', 'tuberlin'],
                        help='Dataset name')
    parser.add_argument('--num_sample', default=50, type=int, help='Vis sample number')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    args = parser.parse_args()
    data_root, data_name, num_sample, save_root = args.data_root, args.data_name, args.num_sample, args.save_root
    random.seed(1)

    old_paths, new_paths, photo_paths = [], [], []
    for class_name in sorted(glob.glob('{}/{}/train/sketch/*'.format(data_root, data_name))):
        class_name = class_name.split('/')[-1]
        olds = sorted(glob.glob('{}/{}/train/sketch/{}/*_real.png'.format(data_root, data_name, class_name)))
        old_paths += random.choices(olds, k=num_sample)
        # random.sample(olds, k=num_sample)
        news = sorted(glob.glob('{}/{}/train/sketch/{}/*_fake.png'.format(data_root, data_name, class_name)))
        new_paths += random.choices(news, k=num_sample)
        photos = sorted(glob.glob('{}/{}/train/photo/{}/*_real.png'.format(data_root, data_name, class_name)))
        photo_paths += random.choices(photos, k=num_sample)

    model = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='max').cuda()
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
                                    transforms.ToTensor()])

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

    ref = np.array([1.0, 0.0], dtype=np.float32)
    old_angles = [int(angle_between(point, ref)) for point in old_sketches]
    new_angles = [int(angle_between(point, ref)) for point in new_sketches]
    photo_angles = [int(angle_between(point, ref)) for point in photos]

    sns.set()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].set_title('Without Domain Elimination')
    axes[0].set_xlabel('Angle')
    axes[0].set_ylabel('# samples')
    axes[1].set_title('With Domain Elimination')
    axes[1].set_xlabel('Angle')
    axes[1].set_ylabel('# samples')
    data = {'sketch': old_angles, 'photo': photo_angles}
    sns.histplot(pd.DataFrame(data), bins=60, palette=['g', 'b'], ax=axes[0])
    data = {'sketch': new_angles, 'photo': photo_angles}
    sns.histplot(pd.DataFrame(data), bins=60, palette=['g', 'b'], ax=axes[1])
    plt.savefig('{}/{}_dist.pdf'.format(save_root, data_name), bbox_inches='tight', pad_inches=0.1)
