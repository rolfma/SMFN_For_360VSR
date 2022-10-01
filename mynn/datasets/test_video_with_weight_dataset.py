import torch
import torch.utils.data
import random
import math
import numpy as np
from pathlib import Path

from torch.utils.data import dataset
from mynn.utils.img_util import rgb2ycbcr
from mynn.utils.registry import DATASET_REGISTRY
from mynn.utils import FileClient, imfrombytes, img2tensor, mod_crop


@DATASET_REGISTRY.register()
class TestVideoWithWeightDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_opt, phase='test'):
        super().__init__()
        color_mode = dataset_opt['color_mode']

        _phase = ['train', 'val', 'test']
        assert phase in _phase, f"Wrong phase {phase}, supported choices are {_phase}"
        _color_mode = ['RGB', 'Y']
        assert color_mode in _color_mode, f"Wrong color_mode {color_mode}, supported choices are {_color_mode}"

        self.dataset_opt = dataset_opt
        self.phase = phase
        self.color_mode = color_mode
        self.file_client = None
        self.num_frame = dataset_opt['num_frame']
        self.gt_root = Path(dataset_opt['gt_root'])
        self.lq_root = Path(dataset_opt['lq_root'])
        self.interval_list = dataset_opt.get('interval_list', [1])
        self.mod_crop = dataset_opt['mod_crop']
        self.scale = dataset_opt['scale']
        self.min_frame_idx = int(dataset_opt['min_frame_idx'])
        self.max_frame_idx = int(dataset_opt['max_frame_idx'])
        self.weights = dict()

        self.keys = []

        with open(dataset_opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend(
                    [f'{folder}/{i:03d}' for i in range(self.min_frame_idx, self.min_frame_idx + int(frame_num))])

        # IO backend.
        # self.io_backend_opt = dataset_opt['io_backend']
        # self.is_lmdb = False

    def get(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient()

        key = self.keys[index]
        clip_name, frame_name = key.split('/')
        center_frame_idx = int(frame_name)

        interval = random.choice(self.interval_list)

        num_half_frame = self.num_frame // 2

        start_frame_idx = center_frame_idx - num_half_frame * interval
        end_frame_idx = center_frame_idx + num_half_frame * interval

        frame_name = f'{center_frame_idx:03d}'
        neighbor_list = [frame_idx for frame_idx in range(start_frame_idx, end_frame_idx + 1, interval)]
        for idx in range(len(neighbor_list)):
            if neighbor_list[idx] < self.min_frame_idx:
                neighbor_list[idx] = self.min_frame_idx
            elif neighbor_list[idx] > self.max_frame_idx:
                neighbor_list[idx] = self.max_frame_idx

        assert len(neighbor_list) == self.num_frame, f'Wrong length of neighbor_list:{len(neighbor_list)}'

        gt_weight = []
        # Read GT frame.
        img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        if self.color_mode == 'Y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)
        if self.mod_crop > 0:
            img_gt = mod_crop(img=img_gt, scale=self.mod_crop)
        gt_weight.append(img_gt)

        # Get weights.
        h, w, c = img_gt.shape
        w_key = '-'.join([str(h), str(w)])
        w_value = self.weights.get(w_key, None)
        if w_value is None:
            equ = np.zeros([h, w])
            for j in range(0, h):  #hang
                for i in range(0, w):  #lie
                    equ[j, i] = math.cos((j - (h / 2) + 0.5) * math.pi / h)
            equ = np.expand_dims(equ, 2)
            self.weights[w_key] = equ
            w_value = equ
        gt_weight.append(w_value)

        # Read neighboring LQ frames.
        img_lqs = []
        for neighbor in neighbor_list:
            img_lq_path = self.lq_root / clip_name / f'{neighbor:03d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            if self.mod_crop > 0:
                img_lq = mod_crop(img=img_lq, scale=self.mod_crop // self.scale)
            if self.color_mode == 'Y':
                img_lq = rgb2ycbcr(img_lq, y_only=True)
            img_lqs.append(img_lq)

        img_lqs.extend(gt_weight)
        img_results = img_lqs

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-2], dim=0)
        img_gt = img_results[-2]
        img_w = img_results[-1]

        return {'lqs': img_lqs, 'gt': img_gt, 'w': img_w, 'key': key}

    def __len__(self):
        return len(self.keys)
