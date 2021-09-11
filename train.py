import argparse
import itertools
import os
import random

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from torch import nn
from torch.backends import cudnn
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Extractor, Encoder, Generator
from utils import DomainDataset, compute_metric

# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(backbone, data_loader, train_optimizer):
    backbone.train()
    sketch_shape_encoder.train()
    photo_shape_encoder.train()
    photo_appearance_encoder.train()
    photo_generator.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for sketch, photo, label in train_bar:
        sketch_proj = backbone(sketch.cuda())
        photo_proj = backbone(photo.cuda())
        sketch_shape = sketch_shape_encoder(sketch_proj)
        photo_shape = photo_shape_encoder(photo_proj)
        photo_appearance = photo_appearance_encoder(photo_proj)
        sketch_generated = photo_generator(torch.cat((sketch_shape, photo_appearance), dim=-1))
        photo_generated = photo_generator(torch.cat((photo_shape, photo_appearance), dim=-1))
        class_loss = class_criterion(sketch_shape, label.cuda()) + class_criterion(photo_shape, label.cuda())
        mse_loss = mse_criterion(sketch_generated, photo) + mse_criterion(photo_generated, photo)
        loss = class_loss + mse_loss
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        total_num += sketch.size(0)
        total_loss += loss.item() * sketch.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# val for one epoch
def val(backbone, sketch_encoder, photo_encoder, data_loader):
    backbone.eval()
    sketch_encoder.eval()
    photo_encoder.eval()
    vectors, domains, labels = [], [], []
    with torch.no_grad():
        for img, domain, label in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            proj = backbone(img.cuda())
            photo = proj[domain == 0]
            sketch = proj[domain == 1]
            photo_emb = photo_encoder(photo)
            sketch_emb = sketch_encoder(sketch)
            emb = torch.cat((photo_emb, sketch_emb), dim=0)
            vectors.append(emb.cpu())
            photo_label = label[domain == 0]
            sketch_label = label[domain == 1]
            label = torch.cat((photo_label, sketch_label), dim=0)
            labels.append(label)
            photo_domain = domain[domain == 0]
            sketch_domain = domain[domain == 1]
            domain = torch.cat((photo_domain, sketch_domain), dim=0)
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
    train_loader = DataLoader(train_data, batch_size=batch_size // 2, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # model define
    extractor = Extractor(backbone_type).cuda()
    sketch_shape_encoder = Encoder(2048 if backbone_type == 'resnet50' else 512, emb_dim).cuda()
    photo_shape_encoder = Encoder(2048 if backbone_type == 'resnet50' else 512, emb_dim).cuda()
    photo_appearance_encoder = Encoder(2048 if backbone_type == 'resnet50' else 512, emb_dim).cuda()
    photo_generator = Generator(2 * emb_dim).cuda()

    # loss setup
    class_criterion = NormalizedSoftmaxLoss(len(train_data.classes), emb_dim).cuda()
    mse_criterion = nn.MSELoss().cuda()
    # optimizer config
    optimizer = AdamW([{'params': extractor.parameters(), 'lr': 1e-5},
                       {'params': class_criterion.parameters(), 'lr': 1e-1},
                       {'params': itertools.chain(sketch_shape_encoder.parameters(), photo_shape_encoder.parameters(),
                                                  photo_appearance_encoder.parameters(), photo_generator.parameters())}
                       ], lr=1e-3, weight_decay=5e-4)
    # training loop
    results = {'train_loss': [], 'val_precise': [], 'P@100': [], 'P@200': [], 'mAP@200': [], 'mAP@all': []}
    save_name_pre = '{}_{}_{}'.format(data_name, backbone_type, emb_dim)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_precise = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(extractor, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        val_precise, features = val(extractor, sketch_shape_encoder, photo_shape_encoder, val_loader)
        results['val_precise'].append(val_precise * 100)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')

        if val_precise > best_precise:
            best_precise = val_precise
            torch.save(extractor.state_dict(), '{}/{}_extractor.pth'.format(save_root, save_name_pre))
            torch.save(sketch_shape_encoder.state_dict(),
                       '{}/{}_sketch_shape_encoder.pth'.format(save_root, save_name_pre))
            torch.save(photo_shape_encoder.state_dict(),
                       '{}/{}_photo_shape_encoder.pth'.format(save_root, save_name_pre))
            torch.save(photo_appearance_encoder.state_dict(),
                       '{}/{}_photo_appearance_encoder.pth'.format(save_root, save_name_pre))
            torch.save(photo_generator.state_dict(), '{}/{}_photo_generator.pth'.format(save_root, save_name_pre))
            torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
