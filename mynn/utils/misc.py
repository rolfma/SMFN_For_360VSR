import time
import torch
import os
import os.path as osp
from queue import Queue
import os
from os import path as osp, read
from pathlib import Path
from mynn.utils.dist_util import master_only

from mynn.utils.logger import get_root_logger, log_print


def count_subdir(path):
    count = 0
    item_list = os.listdir(path)
    for item in item_list:
        item_path = osp.join(path, item)
        if osp.isdir(item_path):
            count += 1
    return count


def scandir(root_path):
    """This function could scan a directory, and return its leaf node subdirectories.

    Args:
        root_path (str): the directory which will be scanned.

    Returns:
        list: leaf node subdirectories.
    """
    leaf_list = set()
    queue = Queue()
    if count_subdir(root_path) > 0:
        queue.put(root_path)

    while not queue.empty():
        subdir = queue.get()
        for ssdir in os.listdir(subdir):
            ssdir = osp.join(subdir, ssdir)
            if osp.isdir(ssdir) and count_subdir(ssdir) > 0:
                queue.put(ssdir)
            elif count_subdir(ssdir) == 0:
                leaf_list.add(ssdir)
    return sorted(list(leaf_list))


# For testing.
def load_model(opt=None, model=None, name='model'):
    exp_root = Path('./experiments')
    exp_name = opt['exp_name']
    dir_path = exp_root / exp_name / 'checkpoints' / name
    current_iter = None
    device=opt['test']['device']
    # device = torch.device(
    #     'cpu') if opt['test']['cuda'] == False else torch.device('cuda')
    if opt['test']['resume_path'] == 'auto':
        # Load checkpoint.
        file_path = dir_path / f'{exp_name}_latest.pth'
        if device == torch.device('cpu'):
            model.load_state_dict({
                k.replace('module.', ''): v
                for k, v in torch.load(file_path).items()
            })
        else:
            model.load_state_dict(torch.load(file_path, map_location=device))
            model = torch.nn.DataParallel(model)
        # Get current_iter.
        latest_info = dir_path / f'{exp_name}_latest_iter.txt'
        with open(latest_info, 'r') as fin:
            info = fin.readline()
            current_iter = int(info)
        file_path = dir_path / f'{exp_name}_{current_iter}.pth'
    else:
        file_path = opt['test']['resume_path']

        if device == torch.device('cpu'):
            model.load_state_dict({
                k.replace('module.', ''): v
                for k, v in torch.load(file_path).items()
            })
        else:
            model.load_state_dict(torch.load(file_path, map_location=device))
    log_print(f'Loaded checkpoint: {file_path}')
    return model


# For training.
def load_checkpoint(exp_name, model, name='model', path=None):
    exp_root = Path('./experiments')
    dir_path = exp_root / exp_name / 'checkpoints' / name
    current_iter = None
    if path is None:
        # Load checkpoint.
        file_path = dir_path / f'{name}_latest.pth'
        model.load_state_dict(torch.load(file_path))
        # Get current_iter.
        latest_info = dir_path / f'{name}_latest_iter.txt'
        with open(latest_info, 'r') as fin:
            info = fin.readline()
            current_iter = int(info)
        file_path = dir_path / f'{name}_{current_iter}.pth'
    else:
        file_path = path
        model.load_state_dict(torch.load(file_path))
        info = file_path.split('_')[-1]
        current_iter = int(info)
    log_print(f'Loaded checkpoint: {file_path}')
    return model, current_iter


def get_time_str():
    return time.strftime('%Y%m%d_%H%M', time.localtime())


@master_only
def save_checkpoint(exp_name, model=None, current_iter=None, name='model'):
    exp_root = Path('./experiments')
    dir_path = exp_root / exp_name / 'checkpoints' / name
    os.makedirs(dir_path, exist_ok=True)

    file_path = dir_path / f'{name}_{current_iter}.pth'
    latest_path = dir_path / f'{name}_latest.pth'
    latest_info = dir_path / f'{name}_latest_iter.txt'
    torch.save(model.state_dict(), file_path)
    torch.save(model.state_dict(), latest_path)
    with open(latest_info, 'w') as fout:
        fout.write(f'{current_iter}')
    log_print(f'Saved checkpoint: {file_path}')
