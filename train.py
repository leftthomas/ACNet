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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Extractor, Discriminator, Generator
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
    sketch_generator.train()
    photo_generator.train()
    sketch_discriminator.train()
    photo_discriminator.train()
    total_extractor_loss, total_generator_loss, total_sketch_discriminator_loss = 0.0, 0.0, 0.0
    total_photo_discriminator_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for sketch, photo, label in train_bar:
        sketch, photo, label = sketch.cuda(), photo.cuda(), label.cuda()

        # generators #
        optimizer_generator.zero_grad()

        fake_sketch = sketch_generator(photo)
        fake_photo = photo_generator(sketch)
        pred_fake_sketch = sketch_discriminator(fake_sketch)
        pred_fake_photo = photo_discriminator(fake_photo)

        # adversarial loss
        target_fake_sketch = torch.ones(pred_fake_sketch.size(), device=pred_fake_sketch.device)
        target_fake_photo = torch.ones(pred_fake_photo.size(), device=pred_fake_photo.device)
        adversarial_loss = adversarial_criterion(pred_fake_sketch, target_fake_sketch) + adversarial_criterion(
            pred_fake_photo, target_fake_photo)
        # cycle loss
        cycle_loss = cycle_criterion(sketch_generator(fake_photo), sketch) + cycle_criterion(
            photo_generator(fake_sketch), photo)

        generators_loss = adversarial_loss + 10 * cycle_loss
        generators_loss.backward()
        optimizer_generator.step()
        total_generator_loss += generators_loss.item() * sketch.size(0)

        # sketch discriminator #
        optimizer_sketch_discriminator.zero_grad()
        pred_real_sketch = sketch_discriminator(sketch)
        target_real_sketch = torch.ones(pred_real_sketch.size(), device=pred_real_sketch.device)
        pred_fake_sketch = sketch_discriminator(fake_sketch.detach())
        target_fake_sketch = torch.zeros(pred_fake_sketch.size(), device=pred_fake_sketch.device)
        adversarial_loss = (adversarial_criterion(pred_real_sketch, target_real_sketch) + adversarial_criterion(
            pred_fake_sketch, target_fake_sketch)) / 2
        adversarial_loss.backward()
        optimizer_sketch_discriminator.step()
        total_sketch_discriminator_loss += adversarial_loss.item() * sketch.size(0)

        # photo discriminator #
        optimizer_photo_discriminator.zero_grad()
        pred_real_photo = photo_discriminator(photo)
        target_real_photo = torch.ones(pred_real_photo.size(), device=pred_real_photo.device)
        pred_fake_photo = photo_discriminator(fake_photo.detach())
        target_fake_photo = torch.zeros(pred_fake_photo.size(), device=pred_fake_photo.device)
        adversarial_loss = (adversarial_criterion(pred_real_photo, target_real_photo) + adversarial_criterion(
            pred_fake_photo, target_fake_photo)) / 2
        adversarial_loss.backward()
        optimizer_photo_discriminator.step()
        total_photo_discriminator_loss += adversarial_loss.item() * photo.size(0)

        # extractor #
        train_optimizer.zero_grad()
        sketch_real_proj = backbone(sketch)
        photo_real_proj = backbone(photo)
        sketch_fake_proj = backbone(fake_sketch.detach())
        photo_fake_proj = backbone(fake_photo.detach())
        loss = class_criterion(sketch_real_proj, label) + class_criterion(photo_real_proj, label) + class_criterion(
            sketch_fake_proj, label) + class_criterion(photo_fake_proj, label)
        loss.backward()
        train_optimizer.step()
        total_extractor_loss += loss.item() * sketch.size(0)

        total_num += sketch.size(0)

        e_loss = total_generator_loss / total_num
        g_loss = total_generator_loss / total_num
        ds_loss = total_sketch_discriminator_loss / total_num
        dp_loss = total_photo_discriminator_loss / total_num
        train_bar.set_description('Train Epoch: [{}/{}] E-Loss: {:.4f} G-Loss: {:.4f} DS-Loss: {:.4f} DP-Loss: {:.4f}'
                                  .format(epoch, epochs, e_loss, g_loss, ds_loss, dp_loss))

    return e_loss, g_loss, ds_loss, dp_loss


# val for one epoch
def val(backbone, sketch_encoder, photo_encoder, data_loader):
    backbone.eval()
    sketch_encoder.eval()
    photo_encoder.eval()
    vectors, domains, labels = [], [], []
    with torch.no_grad():
        for img, domain, label in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            img = img.cuda()
            photo_real = img[domain == 0]
            sketch_real = img[domain == 1]
            if photo_real.size(0) != 0:
                sketch_fake = sketch_encoder(photo_real)
                photo_real_proj = backbone(photo_real)
                sketch_fake_proj = backbone(sketch_fake)
                photo_emb = torch.cat((photo_real_proj, sketch_fake_proj), dim=-1)
            if sketch_real.size(0) != 0:
                photo_fake = photo_encoder(sketch_real)
                sketch_real_proj = backbone(sketch_real)
                photo_fake_proj = backbone(photo_fake)
                sketch_emb = torch.cat((sketch_real_proj, photo_fake_proj), dim=-1)
            if photo_real.size(0) == 0:
                emb = sketch_emb
            if sketch_real.size(0) == 0:
                emb = photo_emb
            if photo_real.size(0) != 0 and sketch_real.size(0) != 0:
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
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=6, type=int, help='Number of epochs over the model to train')
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
    extractor = Extractor(backbone_type, emb_dim // 2).cuda()
    sketch_generator = Generator().cuda()
    photo_generator = Generator().cuda()
    sketch_discriminator = Discriminator().cuda()
    photo_discriminator = Discriminator().cuda()

    # loss setup
    class_criterion = NormalizedSoftmaxLoss(len(train_data.classes), emb_dim // 2).cuda()
    adversarial_criterion = nn.MSELoss()
    cycle_criterion = nn.L1Loss()
    # optimizer config
    optimizer_extractor = AdamW([{'params': extractor.parameters()}, {'params': class_criterion.parameters(),
                                                                      'lr': 1e-1}], lr=1e-5)
    optimizer_generator = AdamW(itertools.chain(sketch_generator.parameters(), photo_generator.parameters()),
                                lr=2e-4, betas=(0.5, 0.999))
    optimizer_sketch_discriminator = AdamW(sketch_discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_photo_discriminator = AdamW(photo_discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    lr_scheduler_generator = LambdaLR(optimizer_generator, lr_lambda=lambda eiter: 1.0 - max(0, eiter - 3) / float(3))
    lr_scheduler_sketch_discriminator = LambdaLR(optimizer_sketch_discriminator,
                                                 lr_lambda=lambda eiter: 1.0 - max(0, eiter - 3) / float(3))
    lr_scheduler_photo_discriminator = LambdaLR(optimizer_photo_discriminator,
                                                lr_lambda=lambda eiter: 1.0 - max(0, eiter - 3) / float(3))
    # training loop
    results = {'extractor_loss': [], 'generator_loss': [], 'sketch_discriminator_loss': [],
               'photo_discriminator_loss': [], 'precise': [], 'P@100': [], 'P@200': [], 'mAP@200': [], 'mAP@all': []}
    save_name_pre = '{}_{}_{}'.format(data_name, backbone_type, emb_dim)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_precise = 0.0
    for epoch in range(1, epochs + 1):
        extractor_loss, generator_loss, sketch_discriminator_loss, photo_discriminator_loss = train(extractor,
                                                                                                    train_loader,
                                                                                                    optimizer_extractor)
        results['extractor_loss'].append(extractor_loss)
        results['generator_loss'].append(generator_loss)
        results['sketch_discriminator_loss'].append(sketch_discriminator_loss)
        results['photo_discriminator_loss'].append(photo_discriminator_loss)
        precise, features = val(extractor, sketch_generator, photo_generator, val_loader)
        results['precise'].append(precise * 100)
        lr_scheduler_generator.step()
        lr_scheduler_sketch_discriminator.step()
        lr_scheduler_photo_discriminator.step()
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')

        if precise > best_precise:
            best_precise = precise
            torch.save(extractor.state_dict(), '{}/{}_extractor.pth'.format(save_root, save_name_pre))
            torch.save(sketch_generator.state_dict(), '{}/{}_sketch_generator.pth'.format(save_root, save_name_pre))
            torch.save(photo_generator.state_dict(), '{}/{}_photo_generator.pth'.format(save_root, save_name_pre))
            torch.save(sketch_discriminator.state_dict(),
                       '{}/{}_sketch_discriminator.pth'.format(save_root, save_name_pre))
            torch.save(photo_discriminator.state_dict(),
                       '{}/{}_photo_discriminator.pth'.format(save_root, save_name_pre))
            torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
