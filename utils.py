import glob
import os

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

normalizer = {'sketchy': [(0.718, 0.705, 0.679), (0.206, 0.203, 0.203)],
              'tuberlin': [(0.493, 0.524, 0.545), (0.222, 0.219, 0.222)]}


def get_transform(data_name, split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(normalizer[data_name][0], normalizer[data_name][1])])
    else:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(normalizer[data_name][0], normalizer[data_name][1])])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, method_name, split='train'):
        super(DomainDataset, self).__init__()

        self.method_name = method_name
        self.images = sorted(glob.glob(os.path.join(data_root, data_name, 'original', split, '*', '*', '*.png')))
        self.transform = get_transform(data_name, split)

        self.labels, self.domains, self.classes = [], [], {}
        i = 0
        for img in self.images:
            domain, label = os.path.dirname(img).split('/')[-2:]
            if label not in self.classes:
                self.classes[label] = i
                i += 1
            self.labels.append(self.classes[label])
            self.domains.append(0 if domain == 'photo' else 1)

    def __getitem__(self, index):
        img_name = self.images[index]
        domain = self.domains[index]
        label = self.labels[index]
        img = Image.open(img_name)
        img_1 = self.transform(img)
        if self.method_name != 'gbd':
            img_2 = self.transform(img)
        else:
            img_2 = self.transform(Image.open(img_name.replace('original', 'generated')))
        return img_1, img_2, domain, label, img_name

    def __len__(self):
        return len(self.images)


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

    acc_a, acc_b = [], []
    for r in ranks:
        correct_a = (torch.eq(b_labels[idx_a[:, 0:r]], a_labels.unsqueeze(dim=-1))).any(dim=-1)
        acc_a.append((torch.sum(correct_a) / correct_a.size(0)).item())
        correct_b = (torch.eq(a_labels[idx_b[:, 0:r]], b_labels.unsqueeze(dim=-1))).any(dim=-1)
        acc_b.append((torch.sum(correct_b) / correct_b.size(0)).item())
    return acc_a, acc_b
