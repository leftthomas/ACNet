import argparse
import glob
import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Image Dist.')
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
        news = sorted(glob.glob('{}/{}/train/sketch/{}/*_fake.png'.format(data_root, data_name, class_name)))
        new_paths += random.choices(news, k=num_sample)
        photos = sorted(glob.glob('{}/{}/train/photo/{}/*_real.png'.format(data_root, data_name, class_name)))
        photo_paths += random.choices(photos, k=num_sample)

    # refactor old path to two stage method
    our_new_paths, our_photo_paths = [], []
    for sketch_path, photo_path in zip(new_paths, photo_paths):
        s_path = sketch_path.replace('/home/rh/Downloads/', 'result/').replace('_fake.png', '.jpg')
        class_name = os.path.dirname(s_path).split('/')[-1]
        s_path = '{}/{}'.format(os.path.dirname(s_path), os.path.basename(s_path)[len(class_name) + 1:])
        our_new_paths.append(s_path)
        p_path = photo_path.replace('/home/rh/Downloads/', '/data/').replace('_real.png', '.jpg')
        class_name = os.path.dirname(p_path).split('/')[-1]
        p_path = '{}/{}'.format(os.path.dirname(p_path), os.path.basename(p_path)[len(class_name) + 1:])
        our_photo_paths.append(p_path)

    new_sketches, photos = [], []
    for paths, embeds in zip([new_paths, photo_paths], [new_sketches, photos]):
        for path in tqdm(paths, desc='processing data'):
            emd = np.array(Image.open(path), dtype=np.float32).flatten()
            embeds.append(emd)
    our_new_sketches, our_photos = [], []
    for paths, embeds in zip([our_new_paths, our_photo_paths], [our_new_sketches, our_photos]):
        for path in tqdm(paths, desc='processing data'):
            emd = np.array(Image.open(path).resize((224, 224), resample=Image.BILINEAR), dtype=np.float32).flatten()
            embeds.append(emd)

    new_sketches = np.stack(new_sketches)
    photos = np.stack(photos)
    our_new_sketches = np.stack(our_new_sketches)
    our_photos = np.stack(our_photos)

    positive_their_distance = (np.abs(new_sketches - photos)).mean(axis=-1)
    positive_our_distance = (np.abs(our_new_sketches - our_photos)).mean(axis=-1)

    # construct negative pairs
    photos = np.split(photos, len(photos) // num_sample, axis=0)
    our_photos = np.split(our_photos, len(our_photos) // num_sample, axis=0)
    indexes, ref = list(range(0, len(photos))), list(range(0, len(photos)))
    while np.any(np.array(indexes) == np.array(ref)):
        random.shuffle(indexes)
    photos = np.concatenate([photos[i] for i in indexes], axis=0)
    our_photos = np.concatenate([our_photos[i] for i in indexes], axis=0)

    negative_their_distance = (np.abs(new_sketches - photos)).mean(axis=-1)
    negative_our_distance = (np.abs(our_new_sketches - our_photos)).mean(axis=-1)

    sns.set()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_title('two-stage training', fontsize=18)
    axes[1].set_title('joint training', fontsize=18)
    data = {'positive': positive_their_distance, 'negative': negative_their_distance}
    sns.histplot(pd.DataFrame(data), palette=['g', 'b'], ax=axes[0], binrange=(0, 250))
    data = {'positive': positive_our_distance, 'negative': negative_our_distance}
    sns.histplot(pd.DataFrame(data), palette=['g', 'b'], ax=axes[1], binrange=(0, 250))
    axes[0].set_xlabel('Distance')
    axes[0].set_ylabel('# pairs')
    axes[1].set_xlabel('Distance')
    axes[1].set_ylabel('# pairs')
    axes[0].set_ylim(0, 600)
    axes[1].set_ylim(0, 600)
    plt.savefig('{}/{}_our_dist.pdf'.format(save_root, data_name), bbox_inches='tight', pad_inches=0.1)
