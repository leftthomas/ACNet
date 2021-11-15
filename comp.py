import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from tqdm import tqdm

from model import Extractor
from utils import get_transform


def draw_fig(vectors, legends, save_path, style=None):
    x_min, x_max = np.min(vectors, 0), np.max(vectors, 0)
    vectors = (vectors - x_min) / (x_max - x_min)
    if style is not None:
        data = pd.DataFrame({'x': vectors[:, 0].tolist(), 'y': vectors[:, 1].tolist(),
                             'label': legends, 'domain': style})
        sns.scatterplot(x='x', y='y', hue='label', style='domain', palette='Set2', data=data)
    else:
        data = pd.DataFrame({'x': vectors[:, 0].tolist(), 'y': vectors[:, 1].tolist(), 'label': legends})
        sns.scatterplot(x='x', y='y', hue='label', palette='Set2', data=data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    custom = [Line2D([], [], marker='.', color='b', linestyle='None'),
              Line2D([], [], marker='.', color='r', linestyle='None')]
    plt.legend(custom, ['Yes', 'No'], loc='lower right')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Embedding Dist.')
    parser.add_argument('--data_root', default='/data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='sketchy', type=str, choices=['sketchy', 'tuberlin'],
                        help='Dataset name')
    parser.add_argument('--num_class', default=5, type=int, help='Vis class number')
    parser.add_argument('--num_sample', default=20, type=int, help='Vis sample number')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')
    parser.add_argument('--load_model', default='result/norm/sketchy_resnet50_512_extractor.pth', type=str,
                        help='Loaded model to vis dist.')

    args = parser.parse_args()
    data_root, data_name, num_class, num_sample = args.data_root, args.data_name, args.num_class, args.num_sample
    save_root, load_model = args.save_root, args.load_model

    model = Extractor(backbone_type='resnet50', emb_dim=512)
    model.load_state_dict(torch.load(load_model, map_location='cpu'))
    model = model.cuda()
    model.eval()

    tsne = TSNE(n_components=2, init='pca', random_state=0)

    sketch_paths, photo_paths, sketch_labels, photo_labels = [], [], [], []
    for class_name in sorted(glob.glob('/data/{}/val/sketch/*'.format(data_name)))[:num_class]:
        class_name = class_name.split('/')[-1]
        sketch_paths += sorted(glob.glob('/data/{}/val/sketch/{}/*'.format(data_name, class_name)))[:num_sample]
        photo_paths += sorted(glob.glob('/data/{}/val/photo/{}/*'.format(data_name, class_name)))[:num_sample]
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
    draw_fig(embeds, labels, '{}/{}_emb.pdf'.format(save_root, data_name), styles)
