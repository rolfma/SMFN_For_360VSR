import os
import os.path as osp
from pathlib import Path

mode = "Image"
gt_folder = Path('data/DIV2KVal/HR')
lq_folder = Path('data/DIV2KVal/LR')
img_list = os.listdir(lq_folder)
for img_name in img_list:
    no_ext, ext = osp.splitext(img_name)
    new_name = no_ext[:-2] + ext
    old_path = lq_folder / img_name
    new_path = lq_folder / new_name
    print(f'{old_path} -->> {new_path}')
    os.rename(old_path, new_path)