import argparse
import glob
import os
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


def draw_fig(vectors, legends, style, ax, text, legend_on=False):
    x_min, x_max = np.min(vectors, 0), np.max(vectors, 0)
    vectors = (vectors - x_min) / (x_max - x_min)
    data = pd.DataFrame({'x': vectors[:, 0].tolist(), 'y': vectors[:, 1].tolist(), '_label': legends, '_domain': style})
    if legend_on:
        sns.scatterplot(x='x', y='y', hue='_label', style='_domain', data=data, ax=ax)
        current_handles, current_labels = ax.get_legend_handles_labels()
        class_handles, class_labels = current_handles[:-2], current_labels[:-2]
        domain_handles, domain_labels = current_handles[-2:], current_labels[-2:]
        legend = ax.legend(domain_handles, domain_labels, loc=2, bbox_to_anchor=(1.05, 0.2), borderaxespad=0.)
        ax.legend(class_handles, class_labels, loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., edgecolor='black')
        ax.add_artist(legend)
    else:
        sns.scatterplot(x='x', y='y', hue='_label', style='_domain', data=data, ax=ax, legend=False)
    ax.text(0.7, 0.97, text, fontsize=12)
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

    our_path = 'result/all/{}_resnet50_512_extractor.pth'.format(data_name)
    our_model = Extractor(backbone_type='resnet50', emb_dim=512)
    our_model.load_state_dict(torch.load(our_path, map_location='cpu'))
    our_model = our_model.cuda()
    our_model.eval()

    random.seed(5)
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    sketch_paths, photo_paths, sketch_labels, photo_labels = [], [], [], []
    classes = random.sample(sorted(glob.glob('/data/{}/val/sketch/*'.format(data_name))), k=num_class)
    for class_name in classes:
        class_name = class_name.split('/')[-1]
        sketches = sorted(glob.glob('/data/{}/val/sketch/{}/*'.format(data_name, class_name)))
        sketch_paths += random.sample(sketches, k=num_sample)
        photos = sorted(glob.glob('/data/{}/val/photo/{}/*'.format(data_name, class_name)))
        photo_paths += random.sample(photos, k=num_sample)
        sketch_labels += [class_name] * num_sample
        photo_labels += [class_name] * num_sample

    sketch_gan_paths, photo_gan_paths = [], []
    for sketch_path, photo_path in zip(sketch_paths, photo_paths):
        s_path = sketch_path.replace('/data/', '/home/rh/Downloads/').replace('.jpg', '_fake.png')
        p_path = photo_path.replace('/data/', '/home/rh/Downloads/').replace('.jpg', '_real.png')
        s_path = '{}/{}_{}'.format(os.path.dirname(s_path), os.path.dirname(s_path).split('/')[-1],
                                   os.path.basename(s_path))
        p_path = '{}/{}_{}'.format(os.path.dirname(p_path), os.path.dirname(p_path).split('/')[-1],
                                   os.path.basename(p_path))
        sketch_gan_paths.append(s_path)
        photo_gan_paths.append(p_path)

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

    our_sketches, our_photos = [], []
    for paths, embeds in zip([sketch_paths, photo_paths], [our_sketches, our_photos]):
        for path in tqdm(paths, desc='processing data'):
            emd = transform(Image.open(path)).unsqueeze(dim=0)
            with torch.no_grad():
                embeds.append(our_model(emd.cuda()).squeeze(dim=0).cpu())

    norm_sketches = torch.stack(norm_sketches)
    norm_photos = torch.stack(norm_photos)
    triplet_sketches = torch.stack(triplet_sketches)
    triplet_photos = torch.stack(triplet_photos)
    norm_gan_sketches = torch.stack(norm_gan_sketches)
    norm_gan_photos = torch.stack(norm_gan_photos)
    triplet_gan_sketches = torch.stack(triplet_gan_sketches)
    triplet_gan_photos = torch.stack(triplet_gan_photos)
    our_sketches = torch.stack(our_sketches)
    our_photos = torch.stack(our_photos)

    norm_sketches = tsne.fit_transform(norm_sketches.numpy())
    norm_photos = tsne.fit_transform(norm_photos.numpy())
    triplet_sketches = tsne.fit_transform(triplet_sketches.numpy())
    triplet_photos = tsne.fit_transform(triplet_photos.numpy())
    norm_gan_sketches = tsne.fit_transform(norm_gan_sketches.numpy())
    norm_gan_photos = tsne.fit_transform(norm_gan_photos.numpy())
    triplet_gan_sketches = tsne.fit_transform(triplet_gan_sketches.numpy())
    triplet_gan_photos = tsne.fit_transform(triplet_gan_photos.numpy())
    our_sketches = tsne.fit_transform(our_sketches.numpy())
    our_photos = tsne.fit_transform(our_photos.numpy())

    norm_embeds = np.concatenate((norm_sketches, norm_photos), axis=0)
    triplet_embeds = np.concatenate((triplet_sketches, triplet_photos), axis=0)
    norm_gan_embeds = np.concatenate((norm_gan_sketches, norm_gan_photos), axis=0)
    triplet_gan_embeds = np.concatenate((triplet_gan_sketches, triplet_gan_photos), axis=0)
    our_embeds = np.concatenate((our_sketches, our_photos), axis=0)

    labels = sketch_labels + photo_labels
    styles = ['sketch'] * num_class * num_sample + ['photo'] * num_class * num_sample

    fig, axes = plt.subplots(1, 5, figsize=(27, 4))
    axes[0].set_title(r'$\mathcal{L}_{triplet}$', fontsize=18)
    axes[1].set_title(r'$\mathcal{L}_{triplet}$', fontsize=18)
    axes[2].set_title(r'$\mathcal{L}_{norm}$', fontsize=18)
    axes[3].set_title(r'$\mathcal{L}_{norm}$', fontsize=18)
    axes[4].set_title(r'$\mathcal{L}_{norm}$', fontsize=18)

    draw_fig(triplet_embeds, labels, styles, axes[0], 'w/o synthesis')
    draw_fig(triplet_gan_embeds, labels, styles, axes[1], 'w/ synthesis')
    draw_fig(norm_embeds, labels, styles, axes[2], 'w/o synthesis')
    draw_fig(norm_gan_embeds, labels, styles, axes[3], 'w/ synthesis')
    draw_fig(our_embeds, labels, styles, axes[4], 'joint-training', legend_on=True)
    plt.savefig('{}/{}_emb.pdf'.format(save_root, data_name), bbox_inches='tight', pad_inches=0.1)
