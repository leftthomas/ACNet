import glob

import numpy as np
import timm
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils import get_transform


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


if __name__ == '__main__':
    data_name = 'sketchy'
    data_root = '/home/rh/Downloads/{}/train'.format(data_name)

    old_paths, new_paths, photo_paths = [], [], []
    for class_name in sorted(glob.glob('{}/sketch/*'.format(data_root))):
        class_name = class_name.split('/')[-1]
        old_paths += sorted(glob.glob('{}/sketch/{}/*_real.png'.format(data_root, class_name)))
        new_paths += sorted(glob.glob('{}/sketch/{}/*_fake.png'.format(data_root, class_name)))
        photo_paths += sorted(glob.glob('{}/photo/{}/*_real.png'.format(data_root, class_name)))
    model = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='max').cuda()
    model.eval()
    transform = get_transform(split='test')

    old_sketches, new_sketches, photos = [], [], []
    for paths, embeds in zip([old_paths, new_paths, photo_paths], [old_sketches, new_sketches, photos]):
        for path in tqdm(paths, desc='processing data'):
            emd = np.array(Image.open(path), dtype=np.float32).flatten()
            embeds.append(emd)

    old_sketches = np.stack(old_sketches)
    new_sketches = np.stack(new_sketches)
    photos = np.stack(photos)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    old_sketches = tsne.fit_transform(old_sketches)
    new_sketches = tsne.fit_transform(new_sketches)
    photos = tsne.fit_transform(photos)

    ref = np.array([1.0, 0.0])
    old_angles, new_angles, photo_angles = [], [], []
    for point in old_sketches:
        old_angles.append(angle_between(point, ref))
    for point in new_sketches:
        new_angles.append(angle_between(point, ref))
    for point in photos:
        photo_angles.append(angle_between(point, ref))
    old_angles, new_angles, photo_angles = np.array(old_angles), np.array(new_angles), np.array(photo_angles)
    print(old_angles.mean())
    print(new_angles.mean())
    print(photo_angles.mean())

    # embeds = np.concatenate((old_sketches, new_sketches, photos))
    # x_min, x_max = np.min(embeds, 0), np.max(embeds, 0)
    # embeds = (embeds - x_min) / (x_max - x_min)
    # labels = ['old'] * len(old_sketches) + ['new'] * len(new_sketches) + ['photo'] * len(photos)
    # data = pd.DataFrame({'x': embeds[:, 0].tolist(), 'y': embeds[:, 1].tolist(), 'domain': labels})
    #
    # sns.scatterplot(x='x', y='y', hue='domain', palette='Set2', alpha=0.4, data=data)
    # plt.savefig('result/{}_dist.pdf'.format(data_name), bbox_inches='tight', pad_inches=0.1)
