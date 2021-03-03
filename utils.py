import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

normalizer = {'cufsf': [(0.537, 0.537, 0.537), (0.193, 0.193, 0.193)],
              'shoe': [(0.195, 0.191, 0.189), (0.062, 0.067, 0.070)],
              'chair': [(0.204, 0.197, 0.195), (0.051, 0.059, 0.063)]}


def get_transform(data_name, split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(normalizer[data_name][0], normalizer[data_name][1])])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(normalizer[data_name][0], normalizer[data_name][1])])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, method_name='daco', split='train'):
        super(DomainDataset, self).__init__()

        self.method_name = method_name

        original_path = os.path.join(data_root, data_name, 'original', split, '*', '*.png')
        self.original_images = glob.glob(original_path)
        self.original_images.sort()

        generated_path = os.path.join(data_root, data_name, 'generated', split, '*', '*.png')
        self.generated_images = glob.glob(generated_path)
        self.generated_images.sort()
        self.transform = get_transform(data_name, split)

        self.domains, categories, category_id = [], {}, 0
        for image_name in self.original_images:
            domain = image_name.split('/')[-2]
            if domain == 'A':
                self.domains.append(0)
            else:
                self.domains.append(1)
            category = image_name.split('/')[-1].split('.')[0].split('_')[0]
            if category not in categories:
                categories[category] = category_id
                category_id += 1
        self.labels = []
        for image_name in self.original_images:
            category = image_name.split('/')[-1].split('.')[0].split('_')[0]
            self.labels.append(categories[category])

    def __getitem__(self, index):
        original_img_name = self.original_images[index]
        original_img = Image.open(original_img_name)
        img_1 = self.transform(original_img)
        if self.method_name != 'daco':
            img_2 = self.transform(original_img)
        else:
            generated_img_name = self.generated_images[index]
            generated_img = Image.open(generated_img_name)
            img_2 = self.transform(generated_img)
        return img_1, img_2

    def __len__(self):
        return len(self.original_images)


def recall(vectors, labels, domains, ranks):
    domains = torch.as_tensor(domains, dtype=torch.bool, device=vectors.device)
    labels = torch.as_tensor(labels, dtype=torch.long, device=vectors.device)
    a_vectors = vectors[~domains]
    b_vectors = vectors[domains]
    a_labels = labels[~domains]
    b_labels = labels[domains]
    # domain a ---> domain b
    sim_a = a_vectors.mm(b_vectors.t())
    idx_a = sim_a.topk(k=ranks[-1], dim=-1, largest=True)[1]
    # domain b ---> domain a
    sim_b = b_vectors.mm(a_vectors.t())
    idx_b = sim_b.topk(k=ranks[-1], dim=-1, largest=True)[1]
    # cross domain
    sim = vectors.mm(vectors.t())
    sim.fill_diagonal_(-np.inf)
    idx = sim.topk(k=ranks[-1], dim=-1, largest=True)[1]

    acc_a, acc_b, acc = [], [], []
    for r in ranks:
        correct_a = (torch.eq(b_labels[idx_a[:, 0:r]], a_labels.unsqueeze(dim=-1))).any(dim=-1)
        acc_a.append((torch.sum(correct_a) / correct_a.size(0)).item())
        correct_b = (torch.eq(a_labels[idx_b[:, 0:r]], b_labels.unsqueeze(dim=-1))).any(dim=-1)
        acc_b.append((torch.sum(correct_b) / correct_b.size(0)).item())
        correct = (torch.eq(labels[idx[:, 0:r]], labels.unsqueeze(dim=-1))).any(dim=-1)
        acc.append((torch.sum(correct) / correct.size(0)).item())
    return acc_a, acc_b, acc
