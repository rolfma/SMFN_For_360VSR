import os
from os import path as osp
from PIL import Image
from mynn.utils import scandir


def is_image(frame_name):
    format_list = ['png', 'jpg', 'jpeg']
    if frame_name.split('.')[-1].lower() in format_list:
        return True
    else:
        return False


def get_img_info(img_path):
    img = Image.open(img_path)
    width, height = img.size
    mode = img.mode
    return width, height, mode


def get_seq_info(seq_path):
    frame_list = os.listdir(seq_path)
    assert len(frame_list) != 0, f'{seq_path} is an empty video sequence.'
    width, height, mode = None, None, None
    for frame in frame_list:
        frame_path = osp.join(seq_path, frame)
        if is_image(frame):
            img = Image.open(frame_path)
            if (width, height, mode) == (None, None, None):
                width, height = img.size
                mode = img.mode
            else:
                assert (width, height) == img.size, f'{seq_path} have some frames with different size.'
                assert mode == img.mode, f'{seq_path} have some frames with different mode.'
    return width, height, len(frame_list), mode


def generate_meta_info():

    mode = "Video"
    gt_folder = 'data/MigVR56789/HR'
    meta_info_file = 'mynn/datasets/meta_info_files/meta_info_MigVR56789.txt'

    assert mode in ["Image", "Video"]
    if mode == "Image":
        with open(meta_info_file, 'w') as f:
            img_list = os.listdir(gt_folder)
            for idx, img_name in enumerate(img_list):
                width, height, mode = get_img_info(osp.join(gt_folder, img_name))

                if mode == 'RGB':
                    n_channel = 3
                elif mode == 'L':
                    n_channel = 1
                else:
                    n_channel = 'unknown'

                info = f'{img_name} ({height},{width},{n_channel})'
                print(idx + 1, info)
                f.write(f'{info}\n')

    elif mode == "Video":
        seq_list = scandir(gt_folder)
        with open(meta_info_file, 'w') as f:
            for idx, seq_path in enumerate(seq_list):

                width, height, frame_num, mode = get_seq_info(seq_path)

                if mode == 'RGB':
                    n_channel = 3
                elif mode == 'L':
                    n_channel = 1
                else:
                    n_channel = 'unknown'

                seq_path = osp.relpath(seq_path, gt_folder)
                info = f'{seq_path} {frame_num} ({height},{width},{n_channel})'
                print(idx + 1, info)
                f.write(f'{info}\n')


if __name__ == '__main__':
    generate_meta_info()
