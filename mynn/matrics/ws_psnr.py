import torch
import math


def genERP(i, j, N):
    val = math.pi / N
    w = math.cos((j - (N / 2) + 0.5) * val)
    return w


def compute_map_ws(img):
    """calculate weights for the sphere, the function provide weighting map for a given video
        :img    the input original video
    """
    equ = torch.zeros((img.shape[0], img.shape[1]))

    for j in range(0, equ.shape[0]):
        for i in range(0, equ.shape[1]):
            equ[j, i] = genERP(i, j, equ.shape[0])

    return equ


def getGlobalWSMSEValue(mx, my, mw=None):

    if mw is None:
        mw = compute_map_ws(mx)

    val = torch.sum(torch.multiply((mx - my)**2, mw))
    den = val / torch.sum(mw)

    return den


def get_ws_psnr(image1, image2, mw=None):
    image1 = image1.squeeze()
    image2 = image2.squeeze()
    image1 = image1 * 255.0
    image1 = torch.clip(image1, 0, 255)
    image2 = image2 * 255.0
    image2 = torch.clip(image2, 0, 255)

    ws_mse = getGlobalWSMSEValue(image1, image2, mw=mw)

    try:
        ws_psnr = 20 * math.log10(255.0 / math.sqrt(ws_mse))
    except ZeroDivisionError:
        ws_psnr = math.inf

    return ws_psnr
