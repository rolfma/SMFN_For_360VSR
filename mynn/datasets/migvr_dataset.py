import torch
import torch.utils.data
import random
from pathlib import Path
from mynn.utils.img_util import rgb2ycbcr
from mynn.utils.registry import DATASET_REGISTRY
from mynn.utils import FileClient, imfrombytes, paired_random_crop, augment, img2tensor


@DATASET_REGISTRY.register()
class MigVRDataset(torch.utils.data.Dataset):
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

        # Read GT frame.
        img_gt_path = self.dataroot_gt / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = rgb2ycbcr(img_rgb=img_gt, y_only=True)

        # Read neighboring LQ frames.
        img_lqs = []
        for neighbor in neighbor_list:
            img_lq_path = self.dataroot_lq / clip_name / f'{neighbor:03d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lq = rgb2ycbcr(img_rgb=img_lq, y_only=True)
            img_lqs.append(img_lq)

        if self.phase == 'train':
            # Randomly crop.
            img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, self.gt_size, self.scale, img_gt_path)

            # Augmentation.
            img_lqs.append(img_gt)
            img_results = augment(img_lqs, self.use_hflip, self.use_rot)
        elif self.phase == 'val':
            img_lqs.append(img_gt)
            img_results = img_lqs

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {'lqs': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)
