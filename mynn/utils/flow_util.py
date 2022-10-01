# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/video/optflow.py  # noqa: E501
import cv2
import numpy as np
import os


def flowread(flow_path, quantize=False, concat_axis=0, *args, **kwargs):
    """Read an optical flow map.

    Args:
        flow_path (ndarray or str): Flow path.
        quantize (bool): whether to read quantized pair, if set to True,
            remaining args will be passed to :func:`dequantize_flow`.
        concat_axis (int): The axis that dx and dy are concatenated,
            can be either 0 or 1. Ignored if quantize is False.

    Returns:
        ndarray: Optical flow represented as a (h, w, 2) numpy array
    """
    if quantize:
        assert concat_axis in [0, 1]
        cat_flow = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        if cat_flow.ndim != 2:
            raise IOError(f'{flow_path} is not a valid quantized flow file, its dimension is {cat_flow.ndim}.')
        assert cat_flow.shape[concat_axis] % 2 == 0
        dx, dy = np.split(cat_flow, 2, axis=concat_axis)
        flow = dequantize_flow(dx, dy, *args, **kwargs)
    else:
        with open(flow_path, 'rb') as f:
            try:
                header = f.read(4).decode('utf-8')
            except Exception:
                raise IOError(f'Invalid flow file: {flow_path}')
            else:
                if header != 'PIEH':
                    raise IOError(f'Invalid flow file: {flow_path}, ' 'header does not contain PIEH')

            w = np.fromfile(f, np.int32, 1).squeeze()
            h = np.fromfile(f, np.int32, 1).squeeze()
            flow = np.fromfile(f, np.float32, w * h * 2).reshape((h, w, 2))

    return flow.astype(np.float32)


def flowwrite(flow, filename, quantize=False, concat_axis=0, *args, **kwargs):
    """Write optical flow to file.

    If the flow is not quantized, it will be saved as a .flo file losslessly,
    otherwise a jpeg image which is lossy but of much smaller size. (dx and dy
    will be concatenated horizontally into a single image if quantize is True.)

    Args:
        flow (ndarray): (h, w, 2) array of optical flow.
        filename (str): Output filepath.
        quantize (bool): Whether to quantize the flow and save it to 2 jpeg
            images. If set to True, remaining args will be passed to
            :func:`quantize_flow`.
        concat_axis (int): The axis that dx and dy are concatenated,
            can be either 0 or 1. Ignored if quantize is False.
    """
    if not quantize:
        with open(filename, 'wb') as f:
            f.write('PIEH'.encode('utf-8'))
            np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
            flow = flow.astype(np.float32)
            flow.tofile(f)
            f.flush()
    else:
        assert concat_axis in [0, 1]
        dx, dy = quantize_flow(flow, *args, **kwargs)
        dxdy = np.concatenate((dx, dy), axis=concat_axis)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, dxdy)


def quantize_flow(flow, max_val=0.02, norm=True):
    """Quantize flow to [0, 255].

    After this step, the size of flow will be much smaller, and can be
    dumped as jpeg images.

    Args:
        flow (ndarray): (h, w, 2) array of optical flow.
        max_val (float): Maximum value of flow, values beyond
                        [-max_val, max_val] will be truncated.
        norm (bool): Whether to divide flow values by image width/height.

    Returns:
        tuple[ndarray]: Quantized dx and dy.
    """
    h, w, _ = flow.shape
    dx = flow[..., 0]
    dy = flow[..., 1]
    if norm:
        dx = dx / w  # avoid inplace operations
        dy = dy / h
    # use 255 levels instead of 256 to make sure 0 is 0 after dequantization.
    flow_comps = [quantize(d, -max_val, max_val, 255, np.uint8) for d in [dx, dy]]
    return tuple(flow_comps)


def dequantize_flow(dx, dy, max_val=0.02, denorm=True):
    """Recover from quantized flow.

    Args:
        dx (ndarray): Quantized dx.
        dy (ndarray): Quantized dy.
        max_val (float): Maximum value used when quantizing.
        denorm (bool): Whether to multiply flow values with width/height.

    Returns:
        ndarray: Dequantized flow.
    """
    assert dx.shape == dy.shape
    assert dx.ndim == 2 or (dx.ndim == 3 and dx.shape[-1] == 1)

    dx, dy = [dequantize(d, -max_val, max_val, 255) for d in [dx, dy]]

    if denorm:
        dx *= dx.shape[1]
        dy *= dx.shape[0]
    flow = np.dstack((dx, dy))
    return flow


def quantize(arr, min_val, max_val, levels, dtype=np.int64):
    """Quantize an array of (-inf, inf) to [0, levels-1].

    Args:
        arr (ndarray): Input array.
        min_val (scalar): Minimum value to be clipped.
        max_val (scalar): Maximum value to be clipped.
        levels (int): Quantization levels.
        dtype (np.type): The type of the quantized array.

    Returns:
        tuple: Quantized array.
    """
    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(f'levels must be a positive integer, but got {levels}')
    if min_val >= max_val:
        raise ValueError(f'min_val ({min_val}) must be smaller than max_val ({max_val})')

    arr = np.clip(arr, min_val, max_val) - min_val
    quantized_arr = np.minimum(np.floor(levels * arr / (max_val - min_val)).astype(dtype), levels - 1)

    return quantized_arr


def dequantize(arr, min_val, max_val, levels, dtype=np.float64):
    """Dequantize an array.

    Args:
        arr (ndarray): Input array.
        min_val (scalar): Minimum value to be clipped.
        max_val (scalar): Maximum value to be clipped.
        levels (int): Quantization levels.
        dtype (np.type): The type of the dequantized array.

    Returns:
        tuple: Dequantized array.
    """
    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(f'levels must be a positive integer, but got {levels}')
    if min_val >= max_val:
        raise ValueError(f'min_val ({min_val}) must be smaller than max_val ({max_val})')

    dequantized_arr = (arr + 0.5).astype(dtype) * (max_val - min_val) / levels + min_val

    return dequantized_arr


# ref: https://github.com/sampepose/flownet2-tf/
# blob/18f87081db44939414fc4a48834f9e0da3e69f4c/src/flowlib.py#L240
# def visulize_flow_file(flow_filename, save_dir=None):
# 	flow_data = readFlow(flow_filename)
# 	img = flow2img(flow_data)
# plt.imshow(img)
# plt.show()
# if save_dir:
# idx = flow_filename.rfind("/") + 1
# plt.imsave(os.path.join(save_dir, "%s-vis.png" % flow_filename[idx:-4]), img)


def flow2img(flow_data):
    """Convert optical flow to image.

    Args:
        flow_data (tensor): (h, w, 2)

    Returns:
        np.array: image
    """
    flow_data = flow_data.to('cpu').detach().numpy()
    u = flow_data[:, :, 0].squeeze()
    v = flow_data[:, :, 1].squeeze()

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
	compute optical flow color map
	:param u: horizontal optical flow
	:param v: vertical optical flow
	:return:
	"""

    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img


def make_color_wheel():
    """
	Generate color wheel according Middlebury color code
	:return: Color wheel
	"""
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += +BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel
