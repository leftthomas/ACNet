import argparse
import glob
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
    parser.add_argument('--num_sample', default=200, type=int, help='Vis sample number')
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

    old_sketches, new_sketches, photos = [], [], []
    for paths, embeds in zip([old_paths, new_paths, photo_paths], [old_sketches, new_sketches, photos]):
        for path in tqdm(paths, desc='processing data'):
            emd = np.array(Image.open(path), dtype=np.float32).flatten()
            embeds.append(emd)

    old_sketches = np.stack(old_sketches)
    new_sketches = np.stack(new_sketches)
    photos = np.stack(photos)

    op_distance = (np.abs(old_sketches - photos)).mean(axis=-1)
    np_distance = (np.abs(new_sketches - photos)).mean(axis=-1)

    sns.set()
    data = {'Without Synthesis': op_distance, 'With Synthesis': np_distance}
    ax = sns.histplot(pd.DataFrame(data), palette=['g', 'b'])
    ax.set_xlabel('Distance')
    ax.set_ylabel('# samples')
    plt.savefig('{}/{}_dist.pdf'.format(save_root, data_name), bbox_inches='tight', pad_inches=0.1)
