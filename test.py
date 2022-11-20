import argparse
import os
import shutil

import torch
from PIL import Image

from model import Extractor, Generator
from utils import DomainDataset, get_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('--data_root', default='/data', type=str, help='Datasets root path')
    parser.add_argument('--query_name', default='/data/sketchy/val/sketch/cow/n01887787_591-14.jpg', type=str,
                        help='Query image name')
    parser.add_argument('--data_base', default='result/sketchy_resnet50_512_vectors.pth', type=str,
                        help='Queried database')
    parser.add_argument('--num', default=5, type=int, help='Retrieval number')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    opt = parser.parse_args()

    data_root, query_name, data_base, retrieval_num = opt.data_root, opt.query_name, opt.data_base, opt.num
    save_root, data_name = opt.save_root, data_base.split('/')[-1].split('_')[0]

    vectors = torch.load(data_base)
    val_data = DomainDataset(data_root, data_name, split='val')
    query_image = Image.open(query_name).convert('RGB').resize((224, 224), resample=Image.BILINEAR)
    generator = Generator(in_channels=8, num_block=8).cuda()
    generator.load_state_dict(torch.load('result/sketchy_resnet50_512_generator.pth'))
    generator.eval()
    extractor = Extractor(backbone_type='resnet50', emb_dim=512).cuda()
    extractor.load_state_dict(torch.load('result/sketchy_resnet50_512_extractor.pth'))
    extractor.eval()
    query_feature = extractor(get_transform('val')(query_image).unsqueeze(dim=0).cuda()).squeeze(dim=0).cpu()

    gallery_images, gallery_labels = [], []
    for i, domain in enumerate(val_data.domains):
        if domain == 0:
            gallery_images.append(val_data.images[i])
            gallery_labels.append(val_data.labels[i])
    gallery_features = vectors[torch.tensor(val_data.domains) == 0]

    sim_matrix = query_feature.unsqueeze(0).mm(gallery_features.t()).squeeze()
    idx = sim_matrix.topk(k=retrieval_num, dim=-1)[1]

    result_path = '{}/{}'.format(save_root, query_name.split('/')[-1].split('.')[0])
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)
    query_image.save('{}/query.jpg'.format(result_path))
    for num, index in enumerate(idx):
        retrieval_image = Image.open(gallery_images[index.item()]).resize((224, 224), resample=Image.BILINEAR)
        retrieval_sim = sim_matrix[index.item()].item()
        retrieval_image.save('{}/retrieval_{}_{}.jpg'.format(result_path, num + 1, '%.4f' % retrieval_sim))
