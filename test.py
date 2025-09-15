import os, cv2, json, csv
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from loss_utils import reconstruct, normalize_map, overlay_and_save, evaluate_pixel_level

def test(model, loader, device, save_dir, use_shared=False):
    model.eval(); model.to(device)
    all_preds, all_masks, img_scores, img_labels, paths = [], [], [], [], []
    vis_dir = os.path.join(save_dir, "sample_segmentations")
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            if use_shared:
                x        = batch['img'].to(device)   
                cls      = batch['class']
                m        = batch['mask'].to(device)
                metal_id = batch['m_id'].to(device)
                pth      = batch['path'].to(device)
                
                xh       = model(x, metal_ids=metal_id)
                
            else:
                x        = batch['img'].to(device)
                cls      = batch['class']
                m        = batch['mask']
                pth      = batch['path'].to(device)
                xh       = model(x)
             

            if isinstance(pth, (tuple, list)):pth = pth[0]

            # residual map
            res = (x - xh).abs().mean(dim=1, keepdim=True)
            res = res.squeeze().cpu().numpy()
            res = cv2.GaussianBlur(res, (0,0), sigmaX=1.0)
            res = normalize_map(res)

            all_preds.append(res.astype(np.float32))
            mask_np = m.squeeze().cpu().numpy().astype(np.uint8)
            all_masks.append(mask_np)
            img_scores.append(float(np.percentile(res, 99)))

            cls = cls if isinstance(cls, str) else cls[0]
            cls = cls.strip().lower()
            img_labels.append(0 if cls == 'good' else 1)
            paths.append(pth)

    pix_auroc, best_dice, best_t = evaluate_pixel_level(all_preds, all_masks)
    img_auroc = roc_auc_score(np.array(img_labels, dtype=np.int32), np.array(img_scores, dtype= float))
    summary = {
        "pixel_AUROC": float(pix_auroc),
        "best_Dice": float(best_dice),
        "best_threshold": float(best_t),
        "image_AUROC": float(img_auroc)}

    #predicted masks
    mask_dir = os.path.join(save_dir, "pred_masks")
    os.makedirs(mask_dir, exist_ok=True)
    for res_np, p in zip(all_preds, paths):
        pb = (res_np >= best_t).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(mask_dir, Path(p).stem + "_predmask.png"), pb)

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
