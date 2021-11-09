import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning.losses import TripletMarginLoss
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from utils import DomainDataset, compute_metric

sys.path.append('..')
from model import Extractor

# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(backbone, data_loader):
    backbone.train()
    total_extractor_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for img, domain, label in train_bar:
        img, label = img.cuda(), label.cuda()
        # extractor #
        optimizer_extractor.zero_grad()
        proj = backbone(img)

        # extractor loss
        class_loss = class_criterion(proj, label)
        total_extractor_loss += class_loss.item() * img.size(0)

        class_loss.backward()
        optimizer_extractor.step()

        total_num += img.size(0)
        e_loss = total_extractor_loss / total_num
        train_bar.set_description('Train Epoch: [{}/{}] E-Loss: {:.4f}'.format(epoch, epochs, e_loss))

    return e_loss


# val for one epoch
def val(backbone, data_loader):
    backbone.eval()
    vectors, domains, labels = [], [], []
    with torch.no_grad():
        for img, domain, label in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            img = img.cuda()
            emb = backbone(img)
            vectors.append(emb.cpu())
            labels.append(label)
            domains.append(domain)
        vectors = torch.cat(vectors, dim=0)
        domains = torch.cat(domains, dim=0)
        labels = torch.cat(labels, dim=0)
        acc = compute_metric(vectors, domains, labels)
        results['P@100'].append(acc['P@100'] * 100)
        results['P@200'].append(acc['P@200'] * 100)
        results['mAP@200'].append(acc['mAP@200'] * 100)
        results['mAP@all'].append(acc['mAP@all'] * 100)
        print('Val Epoch: [{}/{}] | P@100:{:.1f}% | P@200:{:.1f}% | mAP@200:{:.1f}% | mAP@all:{:.1f}%'
              .format(epoch, epochs, acc['P@100'] * 100, acc['P@200'] * 100, acc['mAP@200'] * 100,
                      acc['mAP@all'] * 100))
    return acc['precise'], vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='/data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='sketchy', type=str, choices=['sketchy', 'tuberlin'],
                        help='Dataset name')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'vgg16'],
                        help='Backbone type')
    parser.add_argument('--emb_dim', default=512, type=int, help='Embedding dim')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs over the model to train')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    # args parse
    args = parser.parse_args()
    data_root, data_name, backbone_type, emb_dim = args.data_root, args.data_name, args.backbone_type, args.emb_dim
    batch_size, epochs, save_root = args.batch_size, args.epochs, args.save_root

    # data prepare
    train_data = DomainDataset(data_root, data_name, split='train')
    val_data = DomainDataset(data_root, data_name, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # model define
    extractor = Extractor(backbone_type, emb_dim).cuda()

    # loss setup
    class_criterion = TripletMarginLoss()
    # optimizer config
    optimizer_extractor = Adam(extractor.parameters(), lr=1e-4)

    # training loop
    results = {'extractor_loss': [], 'precise': [], 'P@100': [], 'P@200': [], 'mAP@200': [], 'mAP@all': []}
    save_name_pre = '{}_{}_{}'.format(data_name, backbone_type, emb_dim)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_precise = 0.0
    for epoch in range(1, epochs + 1):
        extractor_loss = train(extractor, train_loader)
        results['extractor_loss'].append(extractor_loss)
        precise, features = val(extractor, val_loader)
        results['precise'].append(precise * 100)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')

        if precise > best_precise:
            best_precise = precise
            torch.save(extractor.state_dict(), '{}/{}_extractor.pth'.format(save_root, save_name_pre))
            torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
