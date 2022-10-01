from pathlib import Path
import torch
import yaml
import numpy as np
import math
import os
from mynn.datasets import build_dataset, build_dataloader
from mynn.matrics.ws_ssim import get_ws_ssim
from mynn.models import build_model
from mynn.matrics import get_ws_psnr
from mynn.utils import get_root_logger
from PIL import Image
from mynn.utils.img_util import rgb2ycbcr, ycbcr2rgb

from mynn.utils.logger import log_print
from mynn.utils.misc import load_model

# Read yaml file.
YAML_PATH = 'options/LWPN/lwpn.yaml'
with open(YAML_PATH, 'r', encoding='utf-8') as f:
    opt = yaml.load(f, Loader=yaml.SafeLoader)

# Set logger.
log_dir = Path('experiments') / opt['exp_name']
os.makedirs(log_dir, exist_ok=True)
log_file = log_dir / f"{opt['test']['log_file']}"
_logger = get_root_logger(log_file=log_file)

# Choose CUDA or CPU.
device = "cuda" if opt['test']['cuda'] else "cpu"

# Set GPU list.
gpu_list = ",".join([str(v) for v in opt['test']['gpu_list']])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
log_print(f'device:{device}')

# Get dataset and dataloader.
test_dataset = build_dataset(dataset_opt=opt['test']['dataset'], phase='test')

test_dataloader = build_dataloader(dataset=test_dataset, opt=opt, phase='test')
# Build model.
model = build_model(opt)
model.to(device)
model = torch.nn.DataParallel(model)
# Load checkpoint.
model = load_model(opt=opt, model=model)

log_print('Testing start.')

model.eval()
total_step = len(test_dataloader)
save_root = Path('./experiments') / \
    opt['exp_name'] / 'results' / opt['test']['dataset']['name']
log_print(f'Save path:{save_root}')

val_ws_psnr = 0
val_ws_ssim = 0
with torch.no_grad():
    for step, data in enumerate(test_dataloader):
        # Unpack data.
        gt = data['gt'][:, 0, :, :].unsqueeze(1)
        lqs = data['lqs'][:, :, 0, :, :].unsqueeze(2).to(device)
        img_bic = data['img_bic']
        key = data['key'][0]

        # Forward propagation.
        sr = model(lqs)
        # Compute ws_psnr.
        ws_psnr = get_ws_psnr(sr.cpu(), gt.cpu())
        val_ws_psnr += ws_psnr
        # Compute ws_ssim.
        ws_ssim = get_ws_ssim(sr.cpu(), gt.cpu())
        val_ws_ssim += ws_ssim

        y = sr.cpu().squeeze().numpy()
        img_bic = img_bic.squeeze().numpy()
        bic_ycbcr = rgb2ycbcr(img_bic)
        cb = bic_ycbcr.squeeze()[:, :, 1]
        cr = bic_ycbcr.squeeze()[:, :, 2]

        ycbcr_img = np.stack([y, cb, cr], axis=2)
        rgb_img = ycbcr2rgb(ycbcr_img)

        # Save result.
        clip_name, image_name = key.split('/')
        save_dir = save_root / clip_name
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / f'{image_name}.png'

        rgb_img = rgb_img * 255
        rgb_img = np.clip(rgb_img, 0, 255).round()
        sr_img = Image.fromarray(rgb_img.astype('uint8'))
        sr_img.save(save_path)
        print(f'{step+1}/{total_step}: {key}')
        # if step > 1:
        #     break

    avg_val_ws_psnr = val_ws_psnr / (step + 1)
    log_print(f'WS_PSNR:{avg_val_ws_psnr:.6f}')
    avg_val_ws_ssim = val_ws_ssim / (step + 1)
    log_print(f'WS_SSIM:{avg_val_ws_ssim:.6f}')
    log_print('Test complete.')
