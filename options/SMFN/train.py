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
from mynn.losses import mse_weight_loss, l1_weight_loss
from mynn.matrics import get_psnr
from mynn.models.lwpn_model import dual_network
from mynn.utils import save_checkpoint, load_checkpoint, get_root_logger
from mynn.utils.dist_util import master_only
from mynn.utils.logger import log_print


def main():
    # Read yaml file.
    YAML_PATH = 'options/SFMN/sfmn.yaml'
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        opt = yaml.load(f, Loader=yaml.SafeLoader)

    if opt['train']['one_through'] == True:
        opt['train']['total_iter'] = 4
        opt['train']['save_freq'] = 4
        opt['train']['val_freq'] = 4
        opt['train']['dis_freq'] = 1

    # Set GPU list.
    gpu_list = ",".join([str(v) for v in opt['train']['gpu_list']])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # Support for single-node multi-GPU data parallel training.
    parser = argparse.ArgumentParser(
        description='Distributed Training Configuration.')
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of node of distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:1124',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend',
                        default='nccl',
                        type=str,
                        help='distributed backend')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    args = parser.parse_args()

    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    ngpus_per_node = len(opt['train']['gpu_list'])
    args.world_size = args.world_size * ngpus_per_node
    mp.spawn(main_worker,
             nprocs=ngpus_per_node,
             args=(ngpus_per_node, args, opt))


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
    train_dataset = build_dataset(dataset_opt=opt['train']['dataset'],
                                  phase='train')
    test_dataset = build_dataset(dataset_opt=opt['test']['dataset'],
                                 phase='test')

    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = build_dataloader(dataset=train_dataset,
                                        opt=opt,
                                        phase='train',
                                        sampler=train_sampler)
    test_dataloader = build_dataloader(dataset=test_dataset,
                                       opt=opt,
                                       phase='test')

    # Get total_iters and total_epochs.
    num_iter_per_epoch = math.ceil(
        len(train_dataset) / opt['train']['batch_size'])
    total_iters = int(opt['train']['total_iter'])
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))

    # Build model.
    model = build_model(opt)
    dual = dual_network()
    model.to(device)
    dual.to(device)
    model = DDP(model,
                device_ids=[gpu],
                broadcast_buffers=False,
                find_unused_parameters=True)
    dual = DDP(dual,
               device_ids=[gpu],
               broadcast_buffers=False,
               find_unused_parameters=True)

    # Build loss function.
    loss_fn = l1_weight_loss()
    log_print(f'Loss: {loss_fn.__class__.__name__}')

    # Get optimizer.
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt['train']['lr'],
                                 betas=opt['train']['betas'])
    optimizer_dual = torch.optim.Adam(dual.parameters(),
                                      lr=opt['train']['lr'],
                                      betas=opt['train']['betas'])

    log_print(f'Optimizer: {optimizer.__class__.__name__}')

    # Load checkpoint.
    current_iter = 0
    current_epoch = 0
    if opt['train']['resume']:
        model, current_iter = load_checkpoint(exp_name=opt['exp_name'],
                                              model=model,
                                              name='model')
        dual, _ = load_checkpoint(exp_name=opt['exp_name'],
                                  model=dual,
                                  name='dual')
        current_epoch = math.floor(current_iter / (num_iter_per_epoch))

    # Learning rate ajusting policy.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=50,
                                                gamma=0.5)
    scheduler_dual = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_dual,
                                                     step_size=50,
                                                     gamma=0.5)

    # Record.
    best_psnr = -math.inf
    losses = 0
    # Training pipline.
    log_print('Training start.')
    start_time = time.time()
    for current_epoch in range(current_epoch, total_epochs + 1):
        for i, data in enumerate(train_dataloader):
            model.train()
            dual.train()

            lqs = data['lqs'].to(device)
            gt = data['gt'].to(device)
            weight = data['w'].to(device)
            key = data['key']

            # Forward propagation.
            sr = model(lqs)
            dual_lr = dual(sr)

            # Compute loss.
            ori_loss = loss_fn(gt, sr, weight)
            dual_loss = loss_fn(dual_lr, lqs[:, 1, :, :, :])
            loss = ori_loss + 0.1 * dual_loss

            losses += loss.item()

            # Update parameters.
            optimizer.zero_grad()
            optimizer_dual.zero_grad()

            loss.backward()

            optimizer.step()
            optimizer_dual.step()

            # Count.
            current_iter += 1

            # Print information.
            if current_iter % opt['train']['dis_freq'] == 0:
                # Get time.
                end_time = time.time()
                used_time = end_time - start_time  # second
                remaining_time = (
                    total_iters - current_iter
                ) / opt['train']['dis_freq'] * used_time / 3600  # hour
                log_print(f"Epoch:{current_epoch} Step:{current_iter} "
                          f"Average loss:{losses/opt['train']['dis_freq']:.6f} "
                          f"Learning Rate:{scheduler.get_last_lr()} "
                          f"Time:{(used_time):.1f} s "
                          f"Remaining Time:{remaining_time:.1f} h")
                # Reset.
                losses = 0
                start_time = time.time()

            # Test.
            if current_iter % opt['train']['val_freq'] == 0:
                log_print('Validation start.')
                with torch.no_grad():
                    model.eval()
                    val_losses = 0
                    val_psnr = 0
                    for step, val_data in enumerate(test_dataloader):
                        lqs = val_data['lqs'].to(device)
                        gt = val_data['gt'].to(device)
                        # mw = val_data['w'].to(device)

                        # Forward propagation.
                        sr = model(lqs)

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
            if current_iter % opt['train'][
                    'save_freq'] == 0 or current_iter >= total_iters:
                save_checkpoint(exp_name=opt['exp_name'],
                                current_iter=current_iter,
                                model=model,
                                name='model')
                save_checkpoint(exp_name=opt['exp_name'],
                                current_iter=current_iter,
                                model=dual,
                                name='dual')

            # End.
            if current_iter >= total_iters:
                log_print('Training complete.')
                return
        scheduler.step()
        scheduler_dual.step()


@master_only
def init_logger(opt):
    # Set logger.
    log_dir = Path('experiments') / opt['exp_name']
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"{opt['train']['log_file']}"
    _logger = get_root_logger(log_file=log_file)


if __name__ == '__main__':
    main()
