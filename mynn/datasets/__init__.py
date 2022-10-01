import os
import os.path as osp
import importlib

from torch.utils.data.dataloader import DataLoader
from mynn.utils.logger import log_print
from mynn.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset', 'build_dataloader']

# Import modules automatically.
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in os.listdir(model_folder) if v.endswith('_dataset.py')]
_model_modules = [importlib.import_module(f'mynn.datasets.{file_name}') for file_name in model_filenames]


def build_dataset(dataset_opt=None, phase='train'):
    _phase = ['train', 'val', 'test']
    assert phase in _phase, f'Wrong phase, supported choices are {_phase}'
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt, phase)

    if phase == 'train':
        log_print(f"Dataset for training: {dataset_opt['name']}")
    elif phase == 'val':
        log_print(f"Dataset for validating: {dataset_opt['val_partition']}")
    elif phase == 'test':
        log_print(f"Dataset for testing: {dataset_opt['name']}")
    return dataset


def build_dataloader(dataset=None, opt=None, phase='train', sampler=None):
    _phase = ['train', 'val', 'test']
    assert phase in _phase, f'Wrong phase, supported choices are {_phase}'

    if phase == 'train':
        dataset_opt = opt['train']['dataset']
        if sampler is not None:
            is_shuffle = False
        else:
            is_shuffle = dataset_opt['shuffle']
        dataloader = DataLoader(dataset=dataset,
                                batch_size=opt['train']['batch_size'],
                                shuffle=is_shuffle,
                                num_workers=dataset_opt['num_workers'],
                                prefetch_factor=4,
                                pin_memory=True,
                                sampler=sampler)
    elif phase == 'val':
        dataset_opt = opt['train']['dataset']
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    elif phase == 'test':
        dataset_opt = opt['test']['dataset']
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    return dataloader
