from pathlib import Path
import torch
import yaml
import time
import numpy as np
import math
import os
from mynn.datasets import build_dataset, build_dataloader
from mynn.models import build_model
from mynn.utils import get_root_logger
from PIL import Image
from mynn.utils.img_util import rgb2ycbcr, ycbcr2rgb
from mynn.matrics import get_psnr, get_ssim, get_ws_psnr, get_ws_ssim

from mynn.utils.logger import log_print
from mynn.utils.misc import load_model

# Read yaml file.
YAML_PATH = 'options/SMFN/smfn.yaml'
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
model = load_model(opt=opt, model=model, name='model')

log_print('Testing start.')

model.eval()
total_step = len(test_dataloader)
save_root = Path('./experiments') / \
    opt['exp_name'] / 'results' / opt['test']['dataset']['name']
log_print(f'Save path:{save_root}')

val_psnr = dict()
val_ssim = dict()
val_ws_psnr = dict()
val_ws_ssim = dict()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
    enable_timing=True)
timings = []

with torch.no_grad():
    for step, data in enumerate(test_dataloader):
        # Unpack data.
        gt = data['gt'][:, 0, :, :].unsqueeze(1)
        lqs = data['lqs'][:, :, 0, :, :].unsqueeze(2).to(device)
        img_bic = data['img_bic']
        key = data['key'][0]
        clip_name, frame_idx = key.split('/')

        if val_psnr.get(clip_name, None) is None and val_ssim.get(
                clip_name, None) is None:
            val_psnr[clip_name] = []
            val_ssim[clip_name] = []
            val_ws_psnr[clip_name] = []
            val_ws_ssim[clip_name] = []

        # Forward propagation.
    
        sr = model(lqs)
        
           
        sr = sr.cpu().squeeze(0).permute(1, 2, 0)
        gt = gt.cpu().squeeze(0).permute(1, 2, 0)

        # Compute psnr.
        psnr = get_psnr(sr.cpu(), gt.cpu())
        val_psnr[clip_name].append(psnr)
        # Compute ssim.
        ssim = get_ssim(sr.cpu(), gt.cpu())
        val_ssim[clip_name].append(ssim)
        # Compute ws_psnr.
        ws_psnr=get_ws_psnr(sr.cpu(), gt.cpu())
        val_ws_psnr[clip_name].append(ws_psnr)
        #Compute ws_ssim.
        ws_ssim=get_ws_ssim(sr.cpu(), gt.cpu())
        val_ws_ssim[clip_name].append(ws_ssim)

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
        rgb_img = np.clip(rgb_img, 0, 255)
        sr_img = Image.fromarray(rgb_img.astype('uint8'))
        sr_img.save(save_path)
        print(f'{step+1}/{total_step}: {key}')
        if opt['train']['one_through'] and step > 1:
            break

    total_avg_psnr = []
    total_avg_ssim = []
    total_avg_ws_psnr=[]
    total_avg_ws_ssim=[]
    for key in val_psnr.keys():
        total_avg_psnr.extend(val_psnr[key])
        avg_val_psnr = np.mean(val_psnr[key])
        log_print(f'{key} PSNR:{avg_val_psnr:.6f}')
    log_print(f'Total Average PSNR:{np.mean(total_avg_psnr):.6f}')
    for key in val_ssim.keys():
        total_avg_ssim.extend(val_ssim[key])
        avg_val_ssim = np.mean(val_ssim[key])
        log_print(f'{key} SSIM:{avg_val_ssim:.6f}')
    log_print(f'Total Average SSIM:{np.mean(total_avg_ssim):.6f}')
    for key in val_ws_psnr.keys():
        total_avg_ws_psnr.extend(val_ws_psnr[key])
        avg_val_ws_psnr = np.mean(val_ws_psnr[key])
        log_print(f'{key} WS PSNR:{avg_val_ws_psnr:.6f}')
    log_print(f'Total Average WS PSNR:{np.mean(total_avg_ws_psnr):.6f}')
    for key in val_ws_ssim.keys():
        total_avg_ws_ssim.extend(val_ws_ssim[key])
        avg_val_ws_ssim = np.mean(val_ws_ssim[key])
        log_print(f'{key} WS SSIM:{avg_val_ws_ssim:.6f}')
    log_print(f'Total Average WS SSIM:{np.mean(total_avg_ws_ssim):.6f}')

    log_print(f'Average Inference Timing:{np.mean(timings)}')

    log_print('Test complete.')
