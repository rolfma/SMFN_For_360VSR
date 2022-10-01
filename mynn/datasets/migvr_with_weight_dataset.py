import torch
import torch.utils.data
import random
import math
import numpy as np
from pathlib import Path
from mynn.utils.img_util import rgb2ycbcr
from mynn.utils.registry import DATASET_REGISTRY
from mynn.utils import FileClient, imfrombytes, paired_random_crop, augment, img2tensor


@DATASET_REGISTRY.register()
class MigVRWithWeightDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_opt, phase='train'):
        super().__init__()

        _phase = ['train', 'val', 'test']
        assert phase in _phase, f"Wrong phase {phase}, supported choices are {_phase}"

        self.dataset_opt = dataset_opt
        self.phase = phase
        self.file_client = None
        self.num_frame = dataset_opt['num_frame']
        self.gt_size = dataset_opt['gt_size']
        self.dataroot_gt = Path(dataset_opt['dataroot_gt'])
        self.dataroot_lq = Path(dataset_opt['dataroot_lq'])
        self.interval_list = dataset_opt.get('interval_list', [1])
        self.scale = dataset_opt['scale']
        self.use_hflip = dataset_opt['use_hflip']
        self.use_rot = dataset_opt['use_rot']
        self.weights = dict()

        self.keys = []

        with open(dataset_opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:03d}' for i in range(1, int(frame_num) + 1)])
                # print(f'{folder}/{1:03d}')
                # exit(0)

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

        while (start_frame_idx < 1) or (end_frame_idx > 100):
            center_frame_idx = random.randint(1, 100)
            start_frame_idx = center_frame_idx - num_half_frame * interval
            end_frame_idx = center_frame_idx + num_half_frame * interval

        frame_name = f'{center_frame_idx:03d}'
        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1, interval))

        assert len(neighbor_list) == self.num_frame, f'Wrong length of neighbor_list:{len(neighbor_list)}'

        gt_weight = []
        # Read GT frame.
        img_gt_path = self.dataroot_gt / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = rgb2ycbcr(img=img_gt, y_only=True)
        gt_weight.append(img_gt)

        # Get weights.
        """
        h, w, c = img_gt.shape
        w_key = '-'.join([str(h), str(w)])
        w_value = self.weights.get(w_key, None)
        if w_value is None:
            equ = np.zeros([h, w])
            total = 0
            for j in range(0, h):  #hang
                for i in range(0, w):  #lie
                    equ[j, i] = math.cos((j - (h / 2) + 0.5) * math.pi / h)
                    total += equ[j, i]
            equ = equ*h/total
            equ = np.expand_dims(equ, 2)
            self.weights[w_key] = equ
            w_value = equ
        gt_weight.append(w_value)
        """
        
        
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
            img_lq_path = self.dataroot_lq / clip_name / f'{neighbor:03d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lq = rgb2ycbcr(img=img_lq, y_only=True)
            img_lqs.append(img_lq)

        if self.phase == 'train':
            # Randomly crop.
            gt_weight, img_lqs = paired_random_crop(gt_weight, img_lqs, self.gt_size, self.scale, img_gt_path)

            # Augmentation.
            img_lqs.extend(gt_weight)
            img_results = augment(img_lqs, self.use_hflip, self.use_rot)
        elif self.phase == 'val':
            img_lqs.append(img_gt)
            img_results = img_lqs

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-2], dim=0)
        img_gt = img_results[-2]
        img_w = img_results[-1]

        return {'lqs': img_lqs, 'gt': img_gt, 'w': img_w, 'key': key}

    def __len__(self):
        return len(self.keys)
