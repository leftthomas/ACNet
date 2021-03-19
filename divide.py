import argparse
import os
import shutil

import torch
from sklearn.cluster import KMeans
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Model
from utils import DomainDataset

parser = argparse.ArgumentParser(description='Divide Dataset')
parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
parser.add_argument('--data_name', default='sketchy', type=str, choices=['sketchy', 'tuberlin'],
                    help='Dataset name')

args = parser.parse_args()
data_root, data_name = args.data_root, args.data_name
train_data = DomainDataset(data_root, data_name, 'other', split='train')
data_loader = DataLoader(train_data, batch_size=32, num_workers=8)
# using VGG16 pretrained on ImageNet to obtain feature vectors
model = Model(128).cuda()
model.eval()

vectors, img_names = [], []
with torch.no_grad():
    for data, _, _, _, img_name in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
        vectors.append(model(data.cuda())[0])
        img_names += img_name
    vectors = torch.cat(vectors, dim=0)

print('running k-means')
vectors = vectors.cpu().numpy()
# using k-means to cluster photo and sketch
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(vectors)
labels = kmeans.predict(vectors)
print('k-means done')

for img_name, label in tqdm(zip(img_names, labels), total=len(img_names), desc='Data dividing', dynamic_ncols=True):
    domain = 'A' if label else 'B'
    dst = img_name.replace('original', domain)
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    shutil.copy(img_name, dst)
