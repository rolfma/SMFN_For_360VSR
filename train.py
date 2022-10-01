from pathlib import Path
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml
import math
import os
import time
from mynn.datasets import build_dataset, build_dataloader
from mynn.models import build_model
from mynn.losses import CharbonnierLoss
from mynn.utils import save_checkpoint, load_checkpint, get_root_logger
from mynn.utils.dist_util import master_only

from mynn.utils.logger import log_print


def main():
    # Read yaml file.
    YAML_PATH = 'options/BasicVSR/basicvsr.yaml'
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        opt = yaml.load(f, Loader=yaml.SafeLoader)

    if opt['train']['one_through'] == True:
        opt['train']['total_iter'] = 8
        opt['train']['save_freq'] = 4
        opt['train']['val_freq'] = 2
        opt['train']['dis_freq'] = 1

    # Set GPU list.
    gpu_list = ",".join([str(v) for v in opt['train']['gpu_list']])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # Support for single-node multi-GPU data parallel training.
    parser = argparse.ArgumentParser(description='Distributed Training Configuration.')
    parser.add_argument('--world-size', default=1, type=int, help='number of node of distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:1123',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    args = parser.parse_args()

    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    ngpus_per_node = len(opt['train']['gpu_list'])
    args.world_size = args.world_size * ngpus_per_node
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, opt))


def main_worker(gpu, ngpus_per_node, args, opt):
    # Init process group.
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)

    init_logger(opt)
    log_print(f"gpu_list:{opt['train']['gpu_list']}")
    # Set GPU for this process.
    torch.cuda.set_device(gpu)
    device = torch.device('cuda', gpu)
    opt['train']['batch_size'] = opt['train']['batch_size'] // ngpus_per_node

    # Get dataset and dataloader.
    train_dataset = build_dataset(dataset_opt=opt['train']['dataset'], phase='train')
    val_dataset = build_dataset(dataset_opt=opt['train']['dataset'], phase='val')

    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = build_dataloader(dataset=train_dataset, opt=opt, phase='train', sampler=train_sampler)
    val_dataloader = build_dataloader(dataset=val_dataset, opt=opt, phase='val')

    # Get total_iters and total_epochs.
    num_iter_per_epoch = math.ceil(len(train_dataset) / opt['train']['batch_size'])
    total_iters = int(opt['train']['total_iter'])
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))

    # Build model.
    model = build_model(opt)
    model.to(device)
    model = DDP(model, device_ids=[gpu])

    # Build loss function.
    loss_fn = CharbonnierLoss(loss_weight=1.0, reduction='mean')
    log_print(f'Loss: {loss_fn.__class__.__name__}')

    # Get optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['train']['lr'], betas=opt['train']['betas'])
    log_print(f'Optimizer: {optimizer.__class__.__name__}')

    # Load checkpoint.
    current_iter = 0
    current_epoch = 0
    if opt['train']['resume']:
        model, current_iter = load_checkpint(opt=opt, model=model)
        current_epoch = math.floor(total_iters / (num_iter_per_epoch))

    # Learning rate ajusting policy.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=opt['train']['total_iter'],
                                                                     T_mult=1,
                                                                     eta_min=1e-7)

    # Record best loss.
    best_loss = math.inf
    losses = 0

    # Training pipline.
    log_print('Training start.')
    start_time = time.time()
    for current_epoch in range(total_epochs):
        for i, data in enumerate(train_dataloader):
            model.train()

            lqs = data['lqs'].to(device)
            gts = data['gts'].to(device)
            key = data['key']

            # Forward propagation.
            sr = model(lqs)

            # Compute loss.
            loss = loss_fn(gts, sr)

            losses += loss.item()

            # Update parameters.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(current_epoch + i / len(train_dataloader))

            # Count.
            current_iter += 1

            # Print information.
            if current_iter % opt['train']['dis_freq'] == 0:
                # Get time.
                end_time = time.time()
                used_time = end_time - start_time  # second
                remaining_time = (total_iters - current_iter) / opt['train']['dis_freq'] * used_time / 3600  # hour
                log_print(
                    f"Epoch:{current_epoch} Step:{current_iter} Average loss:{losses/opt['train']['dis_freq']:.6f} Time:{(used_time):.1f} s Remaining Time:{remaining_time:.1f} h"
                )
                # Reset.
                losses = 0
                start_time = time.time()

            # Validate.
            if current_iter % opt['train']['val_freq'] == 0:
                log_print('Validation start.')
                with torch.no_grad():
                    model.eval()
                    val_losses = 0
                    for step, val_data in enumerate(val_dataloader):
                        lqs = val_data['lqs'].to(device)
                        gts = val_data['gts'].to(device)

                        # Forward propagation.
                        sr = model(lqs)

                        # Compute loss.
                        val_loss = loss_fn(sr, gts)
                        val_losses += val_loss.item()

                        if opt['train']['one_through'] == True and step >= 1:
                            break
                    avg_val_loss = val_losses / (step + 1)
                    log_print('-----Validation Result-----')
                    log_print(f'Average loss: {avg_val_loss:.6f}')
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        save_checkpoint(opt=opt, model=model, current_iter=current_iter)

            # Save checkpoint.
            if current_iter % opt['train']['save_freq'] == 0 or current_iter >= total_iters:
                save_checkpoint(opt=opt, current_iter=current_iter, model=model)

            # End.
            if current_iter >= total_iters:
                log_print('Training complete.')
                break


@master_only
def init_logger(opt):
    # Set logger.
    log_dir = Path('experiments') / opt['exp_name']
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"{opt['train']['log_file']}"
    _logger = get_root_logger(log_file=log_file)


if __name__ == '__main__':
    main()