import torch
import torch.nn.functional as F
import argparse
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import cv2, os
import matplotlib.pyplot as plt

def laplacian_conv(img):
    k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    k = k.repeat(img.shape[1],1,1,1)  # depthwise
    return F.conv2d(img, k, padding=1, groups=img.shape[1])

def recon_loss(x, x_hat, alpha=0.9, beta=0.1):
    l1 = (x - x_hat).abs().mean()
    lx = laplacian_conv(x); lxh = laplacian_conv(x_hat) ##edge consistency
    ledge = (lx - lxh).abs().mean()
    return alpha*l1 + beta*ledge

def get_loss(name):
    if name == "l1_edge":
        return recon_loss


@torch.no_grad()
def reconstruct(model, x):
    xh = model(x) # residual map per-pixel (sum over channels)
    res = (x - xh).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
    return xh, res

def normalize_map(m):
    # m: [H,W], normalize to 0..1 robustly
    lo, hi = np.percentile(m, 1), np.percentile(m, 99)
    if hi <= lo: hi = m.max() + 1e-6
    m = np.clip((m - lo) / (hi - lo + 1e-9), 0, 1)
    return m

def overlay_and_save(img_np, heat_np, out_path):
    hm = (heat_np*255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
    overlay = (0.6*img_np + 0.4*hm_color)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.imsave(out_path, np.clip(overlay,0,1))

def evaluate_pixel_level(all_preds, all_masks):
    y_pred = np.concatenate([p.flatten() for p in all_preds], axis=0)
    y_true = np.concatenate([m.flatten() for m in all_masks], axis=0)
    # AUROC
    
    auroc = roc_auc_score(y_true, y_pred)
  
    # Dice at best threshold (sweep 100 thresholds)
    best_dice, best_t = 0.0, 0.5
    ts = np.linspace(0.0, 1.0, 100)
    eps = 1e-7
    for t in ts:
        dices=[]
        for p,m in zip(all_preds, all_masks):
            pb = (p >= t).astype(np.uint8)
            inter = (pb*m).sum()
            dice = (2*inter) / (pb.sum() + m.sum() + eps)
            dices.append(dice)
        d = float(np.mean(dices))
        if d > best_dice: best_dice, best_t = d, float(t)
    return auroc, best_dice, best_t
