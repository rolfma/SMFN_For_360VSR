import torch
import yaml
from mynn.datasets import build_dataset
from torchvision import transforms
from pathlib import Path

# Read yaml file.
YAML_PATH = 'options/MyExp/myexp.yaml'
with open(YAML_PATH, 'r', encoding='utf-8') as f:
    opt = yaml.load(f, Loader=yaml.SafeLoader)

# Get dataset and dataloader.
train_dataset = build_dataset(dataset_opt=opt['train']['dataset'], phase='train')
data = train_dataset[3000]
lq = data['lq']
edge_lq = data['edge_lq']
gt = data['gt']
edge_gt = data['edge_gt']
print(torch.max(gt), torch.min(gt))
print(torch.max(edge_gt), torch.min(edge_gt))

save_img_path = Path('misc/tmp')

img = transforms.ToPILImage()(lq)
img.save(save_img_path / 'lq.png')

img = transforms.ToPILImage()(gt)
img.save(save_img_path / 'gt.png')

img = transforms.ToPILImage()(edge_lq)
img.save(save_img_path / 'edge_lq.png')

img = transforms.ToPILImage()(edge_gt)
img.save(save_img_path / 'edge_gt.png')