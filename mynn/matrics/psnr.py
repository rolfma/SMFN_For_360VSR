import torch
import math


def get_mse(mx, my):

    val = torch.mean((mx - my)**2)

    return val


def get_psnr(image1, image2):
    image1 = image1.squeeze()
    image2 = image2.squeeze()
    image1 = image1 * 255.0
    image1 = torch.clip(image1, 0, 255)
    image2 = image2 * 255.0
    image2 = torch.clip(image2, 0, 255)

    mse = get_mse(image1, image2)

    try:
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    except ZeroDivisionError:
        psnr = math.inf

    return psnr
