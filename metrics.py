import torch
import numpy as np
from kornia.losses import ssim as dssim
import lpips

loss_fn_alex = lpips.LPIPS(net='vgg') # best forward scores


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    image_gt = image_gt if torch.is_tensor(image_gt) else torch.from_numpy(image_gt)
    image_pred = image_pred.permute(2,1,0).unsqueeze(0)
    image_gt = image_gt.permute(2,1,0).unsqueeze(0)
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]

def normalize_negative_one(array):
    #array = torch.from_numpy(array)
    min_val = np.min(array)
    max_val = np.max(array)

    return 2 * (array - min_val) / (max_val - min_val) - 1


def lpips_score(image_pred, image_gt):
    # LPIPS require images in -1 to 1 range.
    image_gt = image_gt.cpu().numpy() if torch.is_tensor(image_gt) else image_gt
    rgb_lpips = lpips.im2tensor(normalize_negative_one(image_pred.cpu().numpy()))
    gt_lpips = lpips.im2tensor(normalize_negative_one(image_gt))

    return loss_fn_alex(rgb_lpips, gt_lpips)
