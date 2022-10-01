import os
from pathlib import Path


old_path = Path('MIG/MIG/test/VR/newtest')
new_path = Path('MigNewtest')

kind_list = ['HR', 'LR']

for kind in kind_list:
    src_path = old_path / kind
    clip_list = os.listdir(src_path)
    for clip in clip_list:
        clip_path = src_path / clip
        frame_list = os.listdir(clip_path)
        for frame in frame_list:
            frame_path = clip_path / frame
            if kind == 'HR':
                dst_path = new_path / clip / 'truth'
            elif kind == 'LR':
                dst_path = new_path / clip / 'input4'
            os.makedirs(dst_path, exist_ok=True)
            dst_path = dst_path / frame
            os.system(f'copy {frame_path} {dst_path}')
