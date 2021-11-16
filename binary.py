import torch

from utils import DomainDataset, compute_metric

vectors = torch.load('result/all/tuberlin_resnet50_512_vectors.pth')
vectors[vectors > 0] = 1.0
vectors[vectors < 0] = 0.0
val_data = DomainDataset('/data', 'tuberlin', split='val')
result = compute_metric(vectors, torch.tensor(val_data.domains), torch.tensor(val_data.labels))
print(result)
