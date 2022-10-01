import numpy as np
import math
from numpy.ma.core import exp
import torch
import torch.nn.functional as F
from math import exp
import cv2

myfloat = np.float64


def generate_ws(i, j, M, N):
    res = math.cos((i + 0.5 - N / 2) * math.pi / N)
    return res


def estws(map_ssim):
    N, M = map_ssim.shape
    ws_map = np.zeros([N, M])

    for i in range(N):
        for j in range(M):
            ws_map[i][j] = generate_ws(i, j, M, N)

    return ws_map


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map


def get_ws_ssim(img, img2, crop_border=0):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (Tensor): Images with range [0, 1] with order HWC.
        img2 (Tensor): Images with range [0, 1] with order HWC.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.

    Returns:
        float: ssim result.
    """

    img = img.numpy()
    img2 = img2.numpy()
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    img = img * 255
    img2 = img2 * 255
    img = np.clip(img, 0, 255)
    img2 = np.clip(img2, 0, 255)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    ssims = []
    for i in range(img.shape[2]):
        map_ssim = _ssim(img[..., i], img2[..., i])
        ws = estws(map_ssim)

        wsssim = np.sum(map_ssim * ws) / ws.sum()
        ssims.append(wsssim)

    return np.array(ssims).mean()
