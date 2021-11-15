import argparse
import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

from model import Extractor
from utils import get_transform


def draw_fig(vectors, legends, style, ax, legend_on=False):
    x_min, x_max = np.min(vectors, 0), np.max(vectors, 0)
    vectors = (vectors - x_min) / (x_max - x_min)
    data = pd.DataFrame({'x': vectors[:, 0].tolist(), 'y': vectors[:, 1].tolist(), 'label': legends, 'domain': style})
    if legend_on:
        sns.scatterplot(x='x', y='y', hue='label', style='domain', palette='Set2', data=data, ax=ax)
        ax.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    else:
        sns.scatterplot(x='x', y='y', hue='label', style='domain', palette='Set2', data=data, ax=ax, legend=False)
    ax.set(xlabel=None)
    ax.set(ylabel=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Embedding Dist.')
    parser.add_argument('--data_name', default='sketchy', type=str, choices=['sketchy', 'tuberlin'],
                        help='Dataset name')
    parser.add_argument('--num_class', default=10, type=int, help='Vis class number')
    parser.add_argument('--num_sample', default=20, type=int, help='Vis sample number')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    args = parser.parse_args()
    data_name, num_class, num_sample, save_root = args.data_name, args.num_class, args.num_sample, args.save_root

    norm_path = 'result/norm/{}_resnet50_512_extractor.pth'.format(data_name)
    norm_model = Extractor(backbone_type='resnet50', emb_dim=512)
    norm_model.load_state_dict(torch.load(norm_path, map_location='cpu'))
    norm_model = norm_model.cuda()
    norm_model.eval()

    norm_gan_path = 'result/norm_gan/{}_resnet50_512_extractor.pth'.format(data_name)
    norm_gan_model = Extractor(backbone_type='resnet50', emb_dim=512)
    norm_gan_model.load_state_dict(torch.load(norm_gan_path, map_location='cpu'))
    norm_gan_model = norm_gan_model.cuda()
    norm_gan_model.eval()

    triplet_path = 'result/triplet/{}_resnet50_512_extractor.pth'.format(data_name)
    triplet_model = Extractor(backbone_type='resnet50', emb_dim=512)
    triplet_model.load_state_dict(torch.load(triplet_path, map_location='cpu'))
    triplet_model = triplet_model.cuda()
    triplet_model.eval()

    triplet_gan_path = 'result/triplet_gan/{}_resnet50_512_extractor.pth'.format(data_name)
    triplet_gan_model = Extractor(backbone_type='resnet50', emb_dim=512)
    triplet_gan_model.load_state_dict(torch.load(triplet_gan_path, map_location='cpu'))
    triplet_gan_model = triplet_gan_model.cuda()
    triplet_gan_model.eval()

    random.seed(1)
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    sketch_paths, photo_paths, sketch_labels, photo_labels = [], [], [], []
    for class_name in random.sample(sorted(glob.glob('/data/{}/val/sketch/*'.format(data_name))), k=num_class):
        class_name = class_name.split('/')[-1]
        sketches = sorted(glob.glob('/data/{}/val/sketch/{}/*'.format(data_name, class_name)))
        sketch_paths += random.sample(sketches, k=num_sample)
        photos = sorted(glob.glob('/data/{}/val/photo/{}/*'.format(data_name, class_name)))
        photo_paths += random.sample(photos, k=num_sample)
        sketch_labels += [class_name] * num_sample
        photo_labels += [class_name] * num_sample

    sketch_gan_paths, photo_gan_paths, sketch_gan_labels, photo_gan_labels = [], [], [], []
    for class_name in random.sample(sorted(glob.glob('/home/rh/Downloads/{}/val/sketch/*'.format(data_name))),
                                    k=num_class):
        class_name = class_name.split('/')[-1]
        sketches = sorted(glob.glob('/home/rh/Downloads/{}/val/sketch/{}/*_fake.png'.format(data_name, class_name)))
        sketch_gan_paths += random.sample(sketches, k=num_sample)
        photos = sorted(glob.glob('/home/rh/Downloads/{}/val/photo/{}/*_real.png'.format(data_name, class_name)))
        photo_gan_paths += random.sample(photos, k=num_sample)
        sketch_gan_labels += [class_name] * num_sample
        photo_gan_labels += [class_name] * num_sample

    transform = get_transform(split='val')

    norm_sketches, norm_photos = [], []
    for paths, embeds in zip([sketch_paths, photo_paths], [norm_sketches, norm_photos]):
        for path in tqdm(paths, desc='processing data'):
            emd = transform(Image.open(path)).unsqueeze(dim=0)
            with torch.no_grad():
                embeds.append(norm_model(emd.cuda()).squeeze(dim=0).cpu())

    triplet_sketches, triplet_photos = [], []
    for paths, embeds in zip([sketch_paths, photo_paths], [triplet_sketches, triplet_photos]):
        for path in tqdm(paths, desc='processing data'):
            emd = transform(Image.open(path)).unsqueeze(dim=0)
            with torch.no_grad():
                embeds.append(triplet_model(emd.cuda()).squeeze(dim=0).cpu())

    norm_gan_sketches, norm_gan_photos = [], []
    for paths, embeds in zip([sketch_gan_paths, photo_gan_paths], [norm_gan_sketches, norm_gan_photos]):
        for path in tqdm(paths, desc='processing data'):
            emd = transform(Image.open(path)).unsqueeze(dim=0)
            with torch.no_grad():
                embeds.append(norm_gan_model(emd.cuda()).squeeze(dim=0).cpu())

    triplet_gan_sketches, triplet_gan_photos = [], []
    for paths, embeds in zip([sketch_gan_paths, photo_gan_paths], [triplet_gan_sketches, triplet_gan_photos]):
        for path in tqdm(paths, desc='processing data'):
            emd = transform(Image.open(path)).unsqueeze(dim=0)
            with torch.no_grad():
                embeds.append(triplet_gan_model(emd.cuda()).squeeze(dim=0).cpu())

    norm_sketches = torch.stack(norm_sketches)
    norm_photos = torch.stack(norm_photos)
    triplet_sketches = torch.stack(triplet_sketches)
    triplet_photos = torch.stack(triplet_photos)
    norm_gan_sketches = torch.stack(norm_gan_sketches)
    norm_gan_photos = torch.stack(norm_gan_photos)
    triplet_gan_sketches = torch.stack(triplet_gan_sketches)
    triplet_gan_photos = torch.stack(triplet_gan_photos)

    norm_sketches = tsne.fit_transform(norm_sketches.numpy())
    norm_photos = tsne.fit_transform(norm_photos.numpy())
    triplet_sketches = tsne.fit_transform(triplet_sketches.numpy())
    triplet_photos = tsne.fit_transform(triplet_photos.numpy())
    norm_gan_sketches = tsne.fit_transform(norm_gan_sketches.numpy())
    norm_gan_photos = tsne.fit_transform(norm_gan_photos.numpy())
    triplet_gan_sketches = tsne.fit_transform(triplet_gan_sketches.numpy())
    triplet_gan_photos = tsne.fit_transform(triplet_gan_photos.numpy())

    norm_embeds = np.concatenate((norm_sketches, norm_photos), axis=0)
    triplet_embeds = np.concatenate((triplet_sketches, triplet_photos), axis=0)
    norm_gan_embeds = np.concatenate((norm_gan_sketches, norm_gan_photos), axis=0)
    triplet_gan_embeds = np.concatenate((triplet_gan_sketches, triplet_gan_photos), axis=0)

    labels = sketch_labels + photo_labels
    gan_labels = sketch_gan_labels + photo_gan_labels
    styles = ['sketch'] * num_class * num_sample + ['photo'] * num_class * num_sample

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    axes[0].set_title(r'$\mathcal{L}_{triplet}$ without synthesize')
    axes[1].set_title(r'$\mathcal{L}_{triplet}$ with synthesize')
    axes[2].set_title(r'$\mathcal{L}_{norm}$ without synthesize')
    axes[3].set_title(r'$\mathcal{L}_{norm}$ with synthesize')

    draw_fig(triplet_embeds, labels, styles, axes[0])
    draw_fig(triplet_gan_embeds, gan_labels, styles, axes[1])
    draw_fig(norm_embeds, labels, styles, axes[2])
    draw_fig(norm_gan_embeds, gan_labels, styles, axes[3], legend_on=True)
    plt.savefig('{}/{}_emb.pdf'.format(save_root, data_name), bbox_inches='tight', pad_inches=0.1)
