import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Model, SimCLRLoss
from utils import DomainDataset, recall

# for reproducibility
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for img_1, img_2, img_3 in train_bar:
        _, proj_1 = net(img_1.cuda())
        _, proj_2 = net(img_2.cuda())
        if method_name == 'simclr':
            loss = loss_criterion(proj_1, proj_2)
        else:
            _, proj_3 = net(img_3.cuda())
            loss = loss_criterion(proj_1, proj_2, proj_3)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += img_1.size(0)
        total_loss += loss.item() * img_1.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# val for one epoch
def val(net, data_loader):
    net.eval()
    vectors = []
    with torch.no_grad():
        for data, _, _ in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            vectors.append(net(data.cuda())[0])
        vectors = torch.cat(vectors, dim=0)
        labels = data_loader.dataset.labels
        domains = data_loader.dataset.domains
        acc_a, acc_b, acc = recall(vectors, labels, domains, ranks)
        precise = (acc_a[0] + acc_b[0] + acc[0]) / 3
        desc = 'Val Epoch: [{}/{}] '.format(epoch, epochs)
        for i, r in enumerate(ranks):
            results['val_ab_recall@{}'.format(r)].append(acc_a[i] * 100)
            results['val_ba_recall@{}'.format(r)].append(acc_b[i] * 100)
            results['val_cross_recall@{}'.format(r)].append(acc[i] * 100)
        desc += '| (A->B) R@{}:{:.2f}% | '.format(ranks[0], acc_a[0] * 100)
        desc += '(B->A) R@{}:{:.2f}% | '.format(ranks[0], acc_b[0] * 100)
        desc += '(A<->B) R@{}:{:.2f}% | '.format(ranks[0], acc[0] * 100)
        print(desc)
    return precise, vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='cufsf', type=str, choices=['cufsf', 'shoe', 'chair'],
                        help='Dataset name')
    parser.add_argument('--method_name', default='daco', type=str, choices=['daco', 'simclr'], help='Method name')
    parser.add_argument('--proj_dim', default=128, type=int, help='Projected feature dim for computing loss')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--iters', default=40000, type=int, help='Number of bp over the model to train')
    parser.add_argument('--ranks', default='1,2,4,8', type=str, help='Selected recall')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    # args parse
    args = parser.parse_args()
    data_root, data_name, method_name = args.data_root, args.data_name, args.method_name
    proj_dim, temperature, batch_size, iters = args.proj_dim, args.temperature, args.batch_size, args.iters
    save_root, ranks = args.save_root, [int(k) for k in args.ranks.split(',')]

    # data prepare
    train_data = DomainDataset(data_root, data_name, method_name, split='train')
    val_data = DomainDataset(data_root, data_name, method_name, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    # compute the epochs over the dataset
    epochs = iters // (len(train_data) // batch_size + 1)

    # model setup
    model = Model(proj_dim).cuda()
    # optimizer config
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    if method_name == 'simclr':
        loss_criterion = SimCLRLoss(temperature)
    else:
        loss_criterion = DaCoLoss(temperature)

    # training loop
    results = {'train_loss': [], 'val_precise': []}
    for rank in ranks:
        results['val_ab_recall@{}'.format(rank)] = []
        results['val_ba_recall@{}'.format(rank)] = []
        results['val_cross_recall@{}'.format(rank)] = []
    save_name_pre = '{}_{}'.format(data_name, method_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_precise = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        val_precise, features = val(model, val_loader)
        results['val_precise'].append(val_precise * 100)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')

        if val_precise > best_precise:
            best_precise = val_precise
            torch.save(model.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
            torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
