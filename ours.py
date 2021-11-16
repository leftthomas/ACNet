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


def draw_fig(vectors, legends, style, legend_on=False):
    x_min, x_max = np.min(vectors, 0), np.max(vectors, 0)
    vectors = (vectors - x_min) / (x_max - x_min)
    data = pd.DataFrame({'x': vectors[:, 0].tolist(), 'y': vectors[:, 1].tolist(), '_label': legends, '_domain': style})
    if legend_on:
        ax = sns.scatterplot(x='x', y='y', hue='_label', style='_domain', data=data)
        current_handles, current_labels = ax.get_legend_handles_labels()
        class_handles, class_labels = current_handles[:-2], current_labels[:-2]
        domain_handles, domain_labels = current_handles[-2:], current_labels[-2:]
        legend = ax.legend(domain_handles, domain_labels, loc=2, bbox_to_anchor=(1.05, 0.2), borderaxespad=0.)
        ax.legend(class_handles, class_labels, loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0., edgecolor='black')
        ax.add_artist(legend)
    else:
        ax = sns.scatterplot(x='x', y='y', hue='_label', style='_domain', data=data, legend=False)
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    ax.set_title('TU-Berlin Extended', fontsize=18)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Embedding Dist.')
    parser.add_argument('--data_name', default='sketchy', type=str, choices=['sketchy', 'tuberlin'],
                        help='Dataset name')
    parser.add_argument('--num_class', default=10, type=int, help='Vis class number')
    parser.add_argument('--num_sample', default=20, type=int, help='Vis sample number')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    args = parser.parse_args()
    data_name, num_class, num_sample, save_root = args.data_name, args.num_class, args.num_sample, args.save_root

    model_path = 'result/all/{}_resnet50_512_extractor.pth'.format(data_name)
    model = Extractor(backbone_type='resnet50', emb_dim=512)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.cuda()
    model.eval()

    random.seed(5)
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

    transform = get_transform(split='val')

    sketches, photos = [], []
    for paths, embeds in zip([sketch_paths, photo_paths], [sketches, photos]):
        for path in tqdm(paths, desc='processing data'):
            emd = transform(Image.open(path)).unsqueeze(dim=0)
            with torch.no_grad():
                embeds.append(model(emd.cuda()).squeeze(dim=0).cpu())

    sketches = torch.stack(sketches)
    photos = torch.stack(photos)
    sketches = tsne.fit_transform(sketches.numpy())
    photos = tsne.fit_transform(photos.numpy())
    embeds = np.concatenate((sketches, photos), axis=0)

    labels = sketch_labels + photo_labels
    styles = ['sketch'] * num_class * num_sample + ['photo'] * num_class * num_sample

    draw_fig(embeds, labels, styles, legend_on=True)
    plt.savefig('{}/{}_ours.pdf'.format(save_root, data_name), bbox_inches='tight', pad_inches=0.1)
