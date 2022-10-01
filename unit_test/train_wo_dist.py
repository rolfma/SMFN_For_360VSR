from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
import yaml
import math
import os
import time
from mynn.datasets import build_dataset, build_dataloader
from mynn.models import build_model
from mynn.matrics import get_psnr
from mynn.utils import save_checkpoint, load_checkpoint, get_root_logger
from mynn.utils.dist_util import master_only
from mynn.utils.logger import log_print


def main():
    # Read yaml file.
    YAML_PATH = 'options/MyExp/myexp.yaml'
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        opt = yaml.load(f, Loader=yaml.SafeLoader)

    opt['exp_name'] += '_unit_test'
    if opt['train']['one_through'] == True:
        opt['train']['total_iter'] = 4
        opt['train']['save_freq'] = 4
        opt['train']['val_freq'] = 4
        opt['train']['dis_freq'] = 1
        opt['train']['batch_size'] = 2
    gpu = 0
    ngpus_per_node = 1

    # NOTE: Main Code
    init_logger(opt)
    log_print(f"gpu_list:{opt['train']['gpu_list']}")
    # Set GPU for this process.
    torch.cuda.set_device(gpu)
    device = torch.device('cuda', gpu)
    opt['train']['batch_size'] = opt['train']['batch_size'] // ngpus_per_node

    # Get dataset and dataloader.
    train_dataset = build_dataset(dataset_opt=opt['train']['dataset'], phase='train')
    test_dataset = build_dataset(dataset_opt=opt['test']['dataset'], phase='test')

    # NOTE: Remember change sampler to DistributedSampler in train.py
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = build_dataloader(dataset=train_dataset, opt=opt, phase='train', sampler=train_sampler)
    test_dataloader = build_dataloader(dataset=test_dataset, opt=opt, phase='test')

    # Get total_iters and total_epochs.
    num_iter_per_epoch = math.ceil(len(train_dataset) / opt['train']['batch_size'])
    total_iters = int(opt['train']['total_iter'])
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))

    # Build model.
    model = build_model(None, 'PartialResNet')
    model.to(device)
    net_g = build_model(None, 'PartialResNet')
    net_g.to(device)
    net_d = build_model(None, 'VGGStyleDiscriminator128')
    net_d.to(device)
    net_fine = build_model(None, 'FineNet')
    net_fine.to(device)

    # Build loss function.
    loss_fn = nn.MSELoss()
    loss_fn.to(device)
    adv_loss = nn.BCEWithLogitsLoss()
    log_print(f'Loss: {loss_fn.__class__.__name__}')

    # Get optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['train']['lr'], betas=opt['train']['betas'])
    opt_g = torch.optim.Adam(net_g.parameters(), lr=opt['train']['lr'], betas=opt['train']['betas'])
    opt_d = torch.optim.Adam(net_d.parameters(), lr=opt['train']['lr'], betas=opt['train']['betas'])
    opt_fine = torch.optim.Adam(net_fine.parameters(), lr=opt['train']['lr'], betas=opt['train']['betas'])

    log_print(f'Optimizer: {optimizer.__class__.__name__}')

    # Load checkpoint.
    current_iter = 0
    current_epoch = 0
    if opt['train']['resume']:
        model, current_iter = load_checkpoint(opt['exp_name'], model=model, name='model')
        net_g, current_iter = load_checkpoint(opt['exp_name'], model=net_g, name='net_g')
        net_d, current_iter = load_checkpoint(opt['exp_name'], model=net_d, name='net_d')
        net_fine, current_iter = load_checkpoint(opt['exp_name'], model=net_fine, name='net_fine')
        current_epoch = math.floor(current_iter / (num_iter_per_epoch))

    # Learning rate ajusting policy.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.5)
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer=opt_g, step_size=30, gamma=0.5)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer=opt_d, step_size=30, gamma=0.5)
    scheduler_fine = torch.optim.lr_scheduler.StepLR(optimizer=opt_fine, step_size=30, gamma=0.5)

    # Record.
    best_psnr = -math.inf
    losses_g = 0
    losses_d = 0
    # Training pipline.
    log_print('Training start.')
    start_time = time.time()
    for current_epoch in range(current_epoch, total_epochs + 1):
        for i, data in enumerate(train_dataloader):
            model.train()

            lq = data['lq'].to(device)
            mask_lq = data['mask_lq'].to(device)
            gt = data['gt'].to(device)
            mask_gt = data['mask_gt'].to(device)
            key = data['key']

            # Forward propagation.
            optimizer.zero_grad()
            opt_g.zero_grad()
            opt_d.zero_grad()
            opt_fine.zero_grad()

            sr_edge = model(lq, mask_lq) * mask_gt
            sr_plain = net_g(lq, (1 - mask_lq)) * (1 - mask_gt)
            gt_plain = gt * (1 - mask_gt)
            sr=torch.cat([sr_edge, sr_plain], 1)

            sr = net_fine(sr)

            # Update net_g.
            fake_out = net_d(sr_plain)
            real = torch.ones_like(fake_out)

            loss_g = loss_fn(gt, sr) + 0.006 * adv_loss(fake_out, real)

            loss_g.backward()

            # Update net_d.
            fake_out = net_d(sr_plain.detach())
            real_out = net_d(gt_plain)
            fake = torch.zeros_like(real_out)
            fake_loss = adv_loss(fake_out, fake)
            real_loss = adv_loss(real_out, real)

            loss_d = (fake_loss + real_loss) / 2
            loss_d.backward()

            optimizer.step()
            opt_g.step()
            opt_d.step()
            opt_fine.step()

            # Count.
            current_iter += 1
            losses_g += loss_g.item()
            losses_d += loss_d.item()

            # Print information.
            if current_iter % opt['train']['dis_freq'] == 0:
                # Get time.
                end_time = time.time()
                used_time = end_time - start_time  # second
                remaining_time = (total_iters - current_iter) / opt['train']['dis_freq'] * used_time / 3600  # hour
                log_print(f"Epoch:{current_epoch} Step:{current_iter} "
                          f"Average g loss:{losses_g/opt['train']['dis_freq']:.6f} "
                          f"Average d loss:{losses_d/opt['train']['dis_freq']:.6f} "
                          f"Learning Rate:{scheduler.get_last_lr()} "
                          f"Time:{(used_time):.1f} s "
                          f"Remaining Time:{remaining_time:.1f} h")
                # Reset.
                losses_g = 0
                losses_d = 0
                start_time = time.time()

            # Test.
            if current_iter % opt['train']['val_freq'] == 0:
                log_print('Validation start.')
                with torch.no_grad():
                    model.eval()
                    val_losses = 0
                    val_psnr = 0
                    for step, val_data in enumerate(test_dataloader):
                        lq = val_data['lq'].to(device)
                        gt = val_data['gt'].to(device)
                        mask_lq = val_data['mask_lq'].to(device)
                        mask_gt = val_data['mask_gt'].to(device)

                        # Forward propagation.
                        sr = model(lq, mask_lq)

                        # Compute psnr.
                        psnr = get_psnr(sr.squeeze().cpu(), gt.squeeze().cpu())
                        val_psnr += psnr

                        # Compute loss.
                        val_loss = loss_fn(sr, gt)
                        val_losses += val_loss.item()

                        if opt['train']['one_through'] == True and step >= 1:
                            break
                    avg_val_loss = val_losses / (step + 1)
                    avg_val_psnr = val_psnr / (step + 1)
                    if avg_val_psnr > best_psnr:
                        best_psnr = avg_val_psnr
                    log_print('-----Validation Result-----')
                    log_print(f'Average loss: {avg_val_loss:.6f}')
                    log_print(f'Average PSNR:{avg_val_psnr:.6f}')
                    log_print(f'Best PSNR:{best_psnr:.6f}')
                    log_print(f"Learning rate:{scheduler.get_last_lr()}")

            # Save checkpoint.
            if current_iter % opt['train']['save_freq'] == 0 or current_iter >= total_iters:
                save_checkpoint(exp_name=opt['exp_name'], current_iter=current_iter, model=model, name='model')
                save_checkpoint(exp_name=opt['exp_name'], current_iter=current_iter, model=net_g, name='net_g')
                save_checkpoint(exp_name=opt['exp_name'], current_iter=current_iter, model=net_d, name='net_d')
                save_checkpoint(exp_name=opt['exp_name'], current_iter=current_iter, model=net_fine, name='net_fine')

            # End.
            if current_iter >= total_iters:
                log_print('Training complete.')
                return

        scheduler.step()
        scheduler_g.step()
        scheduler_d.step()
        scheduler_fine.step()


@master_only
def init_logger(opt):
    # Set logger.
    log_dir = Path('experiments') / opt['exp_name']
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"{opt['train']['log_file']}"
    _logger = get_root_logger(log_file=log_file)


if __name__ == '__main__':
    main()